from typing import List, Optional, Tuple, Union
from transformers import MPNetPreTrainedModel, MPNetModel
from torch.nn.functional import cosine_similarity
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging
from torch import Tensor
logger = logging.get_logger(__name__)


class EntityRetrieverMPNet(MPNetPreTrainedModel):
    def __init__(self, config, model=None, args=None):
        super().__init__(config)
        if model is None:
            self.model = MPNetModel(config)
        else:
            print("load fine-tuned model as the base model")
            self.model = model    
        self.post_init()
        self.args = args

    def mean_pooling(self, last_hidden_state, attention_mask):
        device = last_hidden_state.device  # Get the device of the last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(device).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        input_ids_candidates: torch.LongTensor = None,
        attention_mask_candidates: Optional[torch.Tensor] = None,    
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode queries
        transformer_outputs_query = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states_query = transformer_outputs_query[0]
        hidden_states_query = self.mean_pooling(hidden_states_query, attention_mask)
        hidden_states_query = F.normalize(hidden_states_query, p=2, dim=1)

        # Encode candidates
        batch_size, num_candidates, seq_len = input_ids_candidates.size()
        input_ids_candidates = input_ids_candidates.view(-1, seq_len)
        attention_mask_candidates = attention_mask_candidates.view(-1, seq_len)

        transformer_outputs_candidates = self.model(
            input_ids=input_ids_candidates,
            attention_mask=attention_mask_candidates,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states_candidates = transformer_outputs_candidates[0]
        hidden_states_candidates = self.mean_pooling(hidden_states_candidates, attention_mask_candidates)
        hidden_states_candidates = F.normalize(hidden_states_candidates, p=2, dim=1)
        hidden_states_candidates = hidden_states_candidates.view(batch_size, num_candidates, -1)

        similarities = torch.matmul(hidden_states_query.unsqueeze(1), hidden_states_candidates.transpose(1, 2)).squeeze(1)
        softmax_similarities = F.softmax(similarities, dim=1) 
        predicted_labels = torch.argmax(softmax_similarities, dim=1)

        if labels is not None:
            similarities = similarities.float()
            labels = labels.to(similarities.device)
            if self.args.loss_type == 'CMRL':    
                labels_ce = torch.argmax(labels, dim=1)  # Convert one-hot labels to class indices
                loss_ce = nn.CrossEntropyLoss()(similarities, labels_ce)
                # Iterate over the batch to compute hard negatives and positive scores
                loss_cl = 0  # Initialize contrastive loss
                for i in range(labels.size(0)):  # Loop over batch size
                    # Positive and negative scores for the current batch
                    positive_scores = similarities[i, labels[i] == 1]  # Extract positive scores for batch i
                    negative_scores = similarities[i, labels[i] == 0]  # Extract negative scores for batch i
                    
                    if len(negative_scores) > 0:  # Only compute if there are negatives
                        # Hard negative mining: Select top-k hardest negatives
                        hard_negatives = torch.topk(negative_scores, min(self.args.top_k, len(negative_scores))).values  # Shape (top_k,)
                        # Compute softmax-weighted contrastive loss
                        differences = hard_negatives.unsqueeze(1) - positive_scores.unsqueeze(0) + self.args.margin  # Shape (top_k, num_positives)
                        exp_differences = torch.exp(differences)
                        softmax_weights = exp_differences / torch.sum(exp_differences, dim=0, keepdim=True)  # Normalize weights
                        loss_cl += torch.sum(softmax_weights * F.relu(differences))  # Accumulate the loss
                # Normalize the contrastive loss across the batch
                loss_cl = loss_cl / labels.size(0)
                # Combine the losses
                loss = self.args.alpha * loss_ce + (1 - self.args.alpha) * loss_cl

            elif self.args.loss_type == 'CEL':
                labels = torch.argmax(labels, dim=1)
                loss = nn.CrossEntropyLoss()(similarities, labels)
            elif self.args.loss_type == 'CL':
                # Contrastive Loss computation
                positive_pairs = labels * torch.pow(similarities, 2)
                negative_pairs = (1 - labels) * torch.pow(F.relu(1 - similarities), 2)
                loss = torch.mean(positive_pairs + negative_pairs)
            elif self.args.loss_type == 'TL':
                margin = 1.0  # Hyperparameter for the margin
                positive_pairs = labels * similarities
                negative_pairs = (1 - labels) * similarities
                # Triplet loss encourages a positive pair to have higher similarity than any negative pair by at least a margin
                loss = torch.mean(F.relu(negative_pairs - positive_pairs + margin))
            elif self.args.loss_type == 'BCE':
                softmax_similarities = F.softmax(similarities, dim=1)  # Normalize across candidates
                loss = nn.BCELoss()(softmax_similarities, labels)
            elif self.args.loss_type == 'HL':
                margin = 1.0  # Hyperparameter for the margin
                positive_pairs = labels * similarities
                negative_pairs = (1 - labels) * similarities
                loss = torch.mean(F.relu(negative_pairs - positive_pairs + margin))
            elif self.args.loss_type == 'RL':
                margin = 1.0
                # Normalize similarities for stability
                softmax_similarities = F.softmax(similarities, dim=1)
                # Define ranking penalties
                positive_scores = (softmax_similarities * labels).sum(dim=1)  # Sum scores for positive candidates
                ranking_penalty = (1 - labels) * softmax_similarities  # Scores for negative candidates
                # Penalize negative scores being higher than positives
                loss = torch.mean(F.relu(ranking_penalty.sum(dim=1) - positive_scores + margin))
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=(similarities,predicted_labels),
            hidden_states=transformer_outputs_query.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs_query.attentions if output_attentions else None,
        )
        