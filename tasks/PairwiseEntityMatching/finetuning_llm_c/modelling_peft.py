# Copyright 2023-present the HuggingFace Inc. team.
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

from __future__ import annotations
#import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
#from typing import Any, Optional, Union

#import packaging.version
import torch
from torch import nn
#import transformers
#from accelerate import dispatch_model, infer_auto_device_map
#from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
#from accelerate.utils import get_balanced_memory
#from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
#from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
#from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
#from transformers.utils import PushToHubMixin

#from . import __version__
from peft.config import PeftConfig
from peft.tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    IA3Model,
    LoHaModel,
    LoKrModel,
    LoraModel,
    MultitaskPromptEmbedding,
    OFTModel,
    PolyModel,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _get_batch_size,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    id_tensor_storage,
    infer_device,
    load_peft_weights,
    set_peft_model_state_dict,
    shift_tokens_right,
)

from transformers import PretrainedConfig, AutoConfig

PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.LOHA: LoHaModel,
    PeftType.LOKR: LoKrModel,
    PeftType.PROMPT_TUNING: PromptEmbedding,
    PeftType.P_TUNING: PromptEncoder,
    PeftType.PREFIX_TUNING: PrefixEncoder,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
    PeftType.IA3: IA3Model,
    PeftType.OFT: OFTModel,
    PeftType.POLY: PolyModel,
}




import importlib
import os
from typing import Optional

from transformers import (
    
    AutoModelForSequenceClassification,
    AutoTokenizer
    
)

from peft.config import PeftConfig
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from peft.utils.constants import TOKENIZER_CONFIG_NAME
from peft.utils.other import check_file_exists_on_hf_hub
'''

from .config import PeftConfig
from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .utils.constants import TOKENIZER_CONFIG_NAME
from .utils.other import check_file_exists_on_hf_hub

'''
class PeftModelForSequenceClassificationCustom(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForSequenceClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "SEQ_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForSequenceClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """
    

    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        print('!!!!!!!!PeftModelForSequenceClassificationCustom')
        print('====model',model)
        super().__init__(model, peft_config, adapter_name)
        print('1model',model)
        print('2model.score',model.score)
        print('3self.base_model',self.base_model)
        

        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})


        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                if peft_config.peft_type == PeftType.POLY:
                    kwargs["task_ids"] = task_ids
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class _BaseAutoPeftModel:
    _target_class = None
    _target_peft_class = None

    def __init__(self, *args, **kwargs):
        # For consistency with transformers: https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/auto/auto_factory.py#L400
        raise EnvironmentError(  # noqa: UP024
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        r"""
        A wrapper around all the preprocessing steps a user needs to perform in order to load a PEFT model. The kwargs
        are passed along to `PeftConfig` that automatically takes care of filtering the kwargs of the Hub methods and
        the config object init.
        """
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path

        task_type = getattr(peft_config, "task_type", None)

        if cls._target_class is not None:
            target_class = cls._target_class
        elif cls._target_class is None and task_type is not None:
            # this is only in the case where we use `AutoPeftModel`
            raise ValueError(
                "Cannot use `AutoPeftModel` with a task type, please use a specific class for your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)"
            )

        if task_type is not None:
            expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
            if cls._target_peft_class.__name__ != expected_target_class.__name__:
                raise ValueError(
                    f"Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {cls._target_peft_class.__name__ }"
                    " make sure that you are loading the correct model for your task type."
                )
        elif task_type is None and getattr(peft_config, "auto_mapping", None) is not None:
            auto_mapping = getattr(peft_config, "auto_mapping", None)
            base_model_class = auto_mapping["base_model_class"]
            parent_library_name = auto_mapping["parent_library"]

            parent_library = importlib.import_module(parent_library_name)
            target_class = getattr(parent_library, base_model_class)
        else:
            raise ValueError(
                "Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type."
            )

        base_model = target_class.from_pretrained(base_model_path, **kwargs)

        tokenizer_exists = False
        if os.path.exists(os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)):
            tokenizer_exists = True
        else:
            token = kwargs.get("token", None)
            if token is None:
                token = kwargs.get("use_auth_token", None)

            tokenizer_exists = check_file_exists_on_hf_hub(
                repo_id=pretrained_model_name_or_path,
                filename=TOKENIZER_CONFIG_NAME,
                revision=kwargs.get("revision", None),
                repo_type=kwargs.get("repo_type", None),
                token=token,
            )

        if tokenizer_exists:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=kwargs.get("trust_remote_code", False)
            )
            base_model.resize_token_embeddings(len(tokenizer))

        return cls._target_peft_class.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            **kwargs,
        )
class _BaseAutoPeftModelCustom:
    _target_class = None
    _target_peft_class = None

    def __init__(self, *args, **kwargs):
        # For consistency with transformers: https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/auto/auto_factory.py#L400
        raise EnvironmentError(  # noqa: UP024
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        r"""
        A wrapper around all the preprocessing steps a user needs to perform in order to load a PEFT model. The kwargs
        are passed along to `PeftConfig` that automatically takes care of filtering the kwargs of the Hub methods and
        the config object init.
        """
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path

        task_type = getattr(peft_config, "task_type", None)

        if cls._target_class is not None:
            target_class = cls._target_class
        elif cls._target_class is None and task_type is not None:
            # this is only in the case where we use `AutoPeftModel`
            raise ValueError(
                "Cannot use `AutoPeftModel` with a task type, please use a specific class for your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)"
            )

        # if task_type is not None:
        #     expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
        #     if cls._target_peft_class.__name__ != expected_target_class.__name__:
        #         raise ValueError(
        #             f"Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {cls._target_peft_class.__name__ }"
        #             " make sure that you are loading the correct model for your task type."
        #         )
        # elif task_type is None and getattr(peft_config, "auto_mapping", None) is not None:
        #     auto_mapping = getattr(peft_config, "auto_mapping", None)
        #     base_model_class = auto_mapping["base_model_class"]
        #     parent_library_name = auto_mapping["parent_library"]

        #     parent_library = importlib.import_module(parent_library_name)
        #     target_class = getattr(parent_library, base_model_class)
        # else:
        #     raise ValueError(
        #         "Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type."
        #     )
        

        base_model = target_class.from_pretrained(base_model_path, **kwargs) ####!
        

        tokenizer_exists = False
        if os.path.exists(os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)):
            tokenizer_exists = True
        else:
            token = kwargs.get("token", None)
            if token is None:
                token = kwargs.get("use_auth_token", None)

            tokenizer_exists = check_file_exists_on_hf_hub(
                repo_id=pretrained_model_name_or_path,
                filename=TOKENIZER_CONFIG_NAME,
                revision=kwargs.get("revision", None),
                repo_type=kwargs.get("repo_type", None),
                token=token,
            )

        if tokenizer_exists:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=kwargs.get("trust_remote_code", False)
            )
            base_model.resize_token_embeddings(len(tokenizer))

        return cls._target_peft_class.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            **kwargs,
        )

class AutoPeftModel(_BaseAutoPeftModel):
    _target_class = None
    _target_peft_class = PeftModel


from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
class AutoModelForSequenceClassificationCustom(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AutoPeftModelForSequenceClassification(_BaseAutoPeftModel):
    _target_class = AutoModelForSequenceClassification
    _target_peft_class = PeftModelForSequenceClassification

class AutoPeftModelForSequenceClassificationCustom(_BaseAutoPeftModelCustom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    _target_class = AutoModelForSequenceClassificationCustom
    _target_peft_class = PeftModelForSequenceClassificationCustom
    

