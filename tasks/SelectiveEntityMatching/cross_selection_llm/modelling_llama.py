from typing import Optional, Tuple, Union
from transformers import LlamaModel, LlamaPreTrainedModel, MistralModel, MistralPreTrainedModel
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

class CustomEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomEncoder, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

class EntityRetrieverMistral(MistralPreTrainedModel):
    def __init__(self, config, model=None, encoder=None, score=None, args=None):
        super().__init__(config)
        if model is None:
            self.model = MistralModel(config)
        else:
            print("load fine-tuned model as the base model")
            self.model = model
        if score is None:
            self.score = nn.Linear(config.hidden_size, 10, bias=False)
        else:
            self.score = score
        if encoder is None:
            self.encoder = CustomEncoder(input_size=config.hidden_size * 11, output_size=config.hidden_size)
        else:
            self.encoder = encoder   
        self.post_init()
        self.args = args
        # self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

         # Encode queries and candidates together
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]  # Shape: [batch_size, seq_len, hidden_size]
        # Find positions of the special token (e.g., <s>)
        special_token_id = self.config.bos_token_id
        special_token_positions = (input_ids == special_token_id)  # Shape: [batch_size, seq_len], Boolean mask
        special_token_indices = special_token_positions.nonzero(as_tuple=False)  # Shape: [num_special_tokens, 2]
        filtered_special_token_indices = []
        batch_size = input_ids.size(0)
        positional_indices = torch.zeros(hidden_states.size(0), hidden_states.size(1), dtype=torch.long, device=hidden_states.device)
        
        for batch_idx in range(batch_size):
            # Get all special token indices for this batch
            example_indices = special_token_indices[special_token_indices[:, 0] == batch_idx, 1]
            filtered_example_indices = example_indices[1:]
            filtered_special_token_indices.append(filtered_example_indices)

            positional_indices[batch_idx, :filtered_example_indices[0]] = 0
            for i in range(len(filtered_example_indices) - 1):
                start_idx = filtered_example_indices[i] + 1
                end_idx = filtered_example_indices[i + 1]
                positional_indices[batch_idx, start_idx:end_idx] = 1
            positional_indices[batch_idx, filtered_example_indices[-1] + 1:] = 1

        query_embeddings = []
        batch_candidate_embeddings = []

        for batch_idx in range(batch_size):
            example_indices = filtered_special_token_indices[batch_idx]
            
            if len(example_indices) == 0:
                raise ValueError(f"No special tokens found for batch index {batch_idx} beyond the initial <s> token.")

            # Extract query embedding (from start to the first special token delimiter)
            query_embedding = hidden_states[batch_idx, :example_indices[0], :]
            query_embedding = query_embedding[-1, :]
            query_embeddings.append(query_embedding)

            candidate_embeddings = []
            # Extract candidate embeddings (each segment between <s> tokens)
            for i in range(len(example_indices) - 1):
                start_idx = example_indices[i] + 1 
                end_idx = example_indices[i + 1]
                candidate_embedding = hidden_states[batch_idx, start_idx:end_idx, :]
                candidate_embedding = candidate_embedding[-1, :]
                candidate_embeddings.append(candidate_embedding)
            start_idx = example_indices[-1] + 1
            candidate_embedding = hidden_states[batch_idx, start_idx:, :]
            candidate_attention_mask = attention_mask[batch_idx, start_idx:]
            actual_length = candidate_attention_mask.sum().item()
            last_candidate_embedding = candidate_embedding[actual_length - 1, :]
            candidate_embeddings.append(last_candidate_embedding)
            candidate_embeddings = torch.stack(candidate_embeddings, dim=0)
            batch_candidate_embeddings.append(candidate_embeddings)

        # Stack all embeddings
        query_embeddings = torch.stack(query_embeddings, dim=0)  # Shape: [batch_size, hidden_size]
        candidate_embeddings = torch.stack(batch_candidate_embeddings, dim=0)  # Shape: [batch_size, num_candidates, hidden_size]
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=2)

        similarities = torch.matmul(query_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)  # Shape: [batch_size, num_candidates]
        softmax_similarities = F.softmax(similarities, dim=1) 
        predicted_labels = torch.argmax(softmax_similarities, dim=1)

        # Calculate the loss if labels are provided
        if labels is not None:
            labels = labels.float().to(similarities.device)
            labels_ce = torch.argmax(labels, dim=1) 
            loss_ce = nn.CrossEntropyLoss()(similarities, labels_ce)
            loss_cl = 0 
            for i in range(labels.size(0)): 
                positive_scores = similarities[i, labels[i] == 1]  
                negative_scores = similarities[i, labels[i] == 0]
                
                if len(negative_scores) > 0:
                    hard_negatives = torch.topk(negative_scores, min(self.args.top_k, len(negative_scores))).values 
                    differences = hard_negatives.unsqueeze(1) - positive_scores.unsqueeze(0) + self.args.margin 
                    exp_differences = torch.exp(differences)
                    softmax_weights = exp_differences / torch.sum(exp_differences, dim=0, keepdim=True)
                    loss_cl += torch.sum(softmax_weights * F.relu(differences))

            loss_cl = loss_cl / labels.size(0)
            loss = self.args.alpha * loss_ce + (1 - self.args.alpha) * loss_cl
        else:
            loss = None

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(softmax_similarities, predicted_labels),
            past_key_values=None,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
        )
        