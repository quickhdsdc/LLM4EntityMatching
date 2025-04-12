from transformers import AutoTokenizer, AutoModel
import torch
from .modelling_plm import EntityRetrieverMPNet

class ModelLoader:
    def __init__(self) -> None:
        print('Loading the model...')
    
    def load_model_from_path_name_version(self, model_root_path, args):
        tokenizer = AutoTokenizer.from_pretrained(model_root_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = EntityRetrieverMPNet.from_pretrained(model_root_path, device_map="cuda" if torch.cuda.is_available() else "cpu")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.args = args
        
        return model, tokenizer
