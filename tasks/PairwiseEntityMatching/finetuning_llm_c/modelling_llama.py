# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import math
import warnings
from typing import List, Optional, Tuple, Union
from transformers.models.llama.configuration_llama import LlamaConfig
#from transformers.models.llama.modelling_llama import LlamaModel
from transformers import LlamaModel, LlamaPreTrainedModel, MistralModel, MistralPreTrainedModel

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config, pooling_type, model=None, score=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_type = pooling_type
        if model is None:
            self.model = LlamaModel(config)
        else:
            print("load fine-tuned model as the base model")
            self.model = model
        if score is None:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        else:
            self.score = score

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def mean_pooling(self, last_hidden_state, attention_mask):
        device = last_hidden_state.device  # Get the device of the last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(device).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.pooling_type == 'orig':
        # original pooling option of LlamaforSequenceClassification: first score, then choose the last hidden layer
            logits = self.score(hidden_states)
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        # first score, then use the mean value, rather than choosing the last hidden layer
        elif self.pooling_type == 'orig_mean':
            logits = self.score(hidden_states)
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1).expand(logits.size())
                attention_mask = attention_mask.to(logits.device)
                masked_logits = logits * attention_mask
                pooled_logits = masked_logits.sum(dim=1) / attention_mask.sum(dim=1)
            else:
                pooled_logits = logits.mean(dim=1)
        else:
            # first do the mean pooling, then calculate score
            if self.pooling_type == 'mean':
                hidden_states = self.mean_pooling(hidden_states, attention_mask)
            # first choose the last hidden layer, then calculate score
            elif self.pooling_type == 'last':
                hidden_states = self.last_token_embedding(hidden_states, attention_mask)
            hidden_states = hidden_states.to(self.score.weight.dtype)
            self.score = self.score.to(hidden_states.device)
            pooled_logits = self.score(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class MistralForSequenceClassification(MistralPreTrainedModel):
    def __init__(self, config, pooling_type, model=None, score=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_type = pooling_type
        if model is None:
            self.model = MistralModel(config)
        else:
            print("load fine-tuned model as the base model")
            self.model = model
        if score is None:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        else:
            self.score = score

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def mean_pooling(self, last_hidden_state, attention_mask):
        device = last_hidden_state.device  # Get the device of the last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(device).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.pooling_type == 'orig':
        # original pooling option of LlamaforSequenceClassification
            logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        else:
            if self.pooling_type == 'mean':
                hidden_states = self.mean_pooling(hidden_states, attention_mask)
            elif self.pooling_type == 'last':
                hidden_states = self.last_token_embedding(hidden_states, attention_mask)
            hidden_states = hidden_states.to(self.score.weight.dtype)
            self.score = self.score.to(hidden_states.device)
            pooled_logits = self.score(hidden_states)


        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )