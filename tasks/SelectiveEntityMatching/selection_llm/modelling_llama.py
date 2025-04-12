from typing import Optional, Tuple, Union
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import MistralModel, MistralPreTrainedModel
from torch.nn.functional import cosine_similarity
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging
logger = logging.get_logger(__name__)


class EntityRetrieverMistral(MistralPreTrainedModel):
    def __init__(self, config, model=None, args=None):
        super().__init__(config)
        if model is None:
            self.model = MistralModel(config)
        else:
            print("load fine-tuned model as the base model")
            self.model = model        
        self.post_init()
        self.args = args

    def last_token_embedding(self, last_hidden_state, attention_mask):
        device = last_hidden_state.device  # Get the device of the last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_indices = sequence_lengths.view(-1, 1).expand(-1, last_hidden_state.size(-1))
        # Ensure that last_token_indices is on the correct device before using it for gather
        last_token_indices = last_token_indices.to(device)
        last_token_embeddings = last_hidden_state.gather(1, last_token_indices.unsqueeze(1)).squeeze(1)
        return last_token_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        input_ids_instr: torch.LongTensor = None,
        attention_mask_instr: Optional[torch.Tensor] = None,
        input_ids_candidates: torch.LongTensor = None,
        attention_mask_candidates: Optional[torch.Tensor] = None,    
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode queries
        transformer_outputs_query = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states_query = transformer_outputs_query[0]
        hidden_states_query = self.last_token_embedding(hidden_states_query, attention_mask)
        hidden_states_query = F.normalize(hidden_states_query, p=2, dim=1)

        # Encode candidates
        batch_size, num_candidates, seq_len = input_ids_candidates.size()
        input_ids_candidates = input_ids_candidates.view(-1, seq_len)
        attention_mask_candidates = attention_mask_candidates.view(-1, seq_len)

        transformer_outputs_candidates = self.model(
            input_ids=input_ids_candidates,
            attention_mask=attention_mask_candidates,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states_candidates = transformer_outputs_candidates[0]
        hidden_states_candidates = self.last_token_embedding(hidden_states_candidates, attention_mask_candidates)
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
                loss = (1 - self.args.alpha) * loss_ce + self.args.alpha * loss_cl
            elif self.args.loss_type == 'InfoNCE':
                # Expect labels as one-hot: shape (batch_size, num_candidates)
                labels_ce = torch.argmax(labels, dim=1)  # Convert to index
                temperature = self.args.temperature if hasattr(self.args, 'temperature') else 0.02
                logits = similarities / temperature
                loss = nn.CrossEntropyLoss()(logits, labels_ce)
            elif self.args.loss_type == 'Focus':
                labels_ce = torch.argmax(labels, dim=1)  # Convert to index
                probs = F.softmax(similarities, dim=1)
                pt = probs[torch.arange(labels.size(0)), labels_ce]  # Get probs for the correct class
                gamma = self.args.focus_gamma if hasattr(self.args, 'focus_gamma') else 2.0
                focus_weights = (1 - pt) ** gamma
                loss_ce = nn.CrossEntropyLoss(reduction='none')(similarities, labels_ce)
                loss = (focus_weights * loss_ce).mean()
        else:
            loss = None

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(similarities,predicted_labels),
            past_key_values=None,
            hidden_states=transformer_outputs_query.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs_query.attentions if output_attentions else None,
        )
        