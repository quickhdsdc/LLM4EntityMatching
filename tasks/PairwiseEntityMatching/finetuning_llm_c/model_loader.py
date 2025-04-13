import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from .modelling_llama import LlamaForSequenceClassification, MistralForSequenceClassification
from dotenv import load_dotenv
import os
load_dotenv()
HF_token = os.getenv("HFTOKEN")


class ModelLoader:
    def __init__(self) -> None:
        print('Loading the model...')
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, # Compute data type for 4-bit base models
            )
        
    
    def load_model_from_path_name_version(self, model_root_path:str, model_name:str, model_version:str, labels, label2id, id2label, pooling_type, device_map:str="auto"):
        tokenizer = AutoTokenizer.from_pretrained(model_root_path, token=HF_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        if 'lama' in model_root_path:
            model = LlamaForSequenceClassification.from_pretrained(model_root_path,
                                            quantization_config = self.bnb_config,
                                            device_map = device_map,
                                            token=HF_token,
                                            num_labels = len(labels),
                                            pooling_type = pooling_type
                                            )
        elif 'Mistral' in model_root_path:
            model = MistralForSequenceClassification.from_pretrained(model_root_path,
                                            quantization_config = self.bnb_config,
                                            device_map = device_map,
                                            token=HF_token,
                                            num_labels = len(labels),
                                            pooling_type = pooling_type
                                            )            
        
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.id2label = id2label
        model.config.label2id = label2id
        if 'Mistral' in model_name:
            model.config.sliding_window = 4096
        return model, tokenizer


def main():
    model_loader = ModelLoader()

if __name__ == "__main__":
    main()