import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import os
load_dotenv()
HF_token = os.getenv("HFTOKEN")

class ModelLoader:
    def __init__(self) -> None:
        '''
        params:task_name: str: name of the task
        params:train_type: str: type of the training data in ['train','train_small']
        params:val_type: str: type of the validation data in ['test']
        params:test_type: str: type of the test data in ['test','test_big']
        '''
        print('Loading the model...')
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, # Compute data type for 4-bit base models
            )
        
    
    def load_model_from_path_name_version(self, model_root_path:str, model_name:str, model_version:str, device_map:str="auto"):
        print('Loading model from...', model_root_path+"/models--"+model_name+"/"+model_version)
        #model, tokenizer = load_model(model_root_path, model_name, model_version, bnb_config, label2id, id2label)	

        tokenizer = AutoTokenizer.from_pretrained(model_root_path,token=HF_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = AutoModelForCausalLM.from_pretrained(model_root_path,token=HF_token,
                                                 quantization_config = self.bnb_config,
                                                 device_map = device_map, 
                                                 ) 
        return model, tokenizer


        

def main():
    model_loader = ModelLoader()

if __name__ == "__main__":
    main()