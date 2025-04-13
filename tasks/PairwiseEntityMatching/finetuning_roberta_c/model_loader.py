import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class ModelLoader:
    def __init__(self) -> None:
        print('Loading the model...')

    def load_model_from_path_name_version(self, model_root_path:str, labels, label2id, id2label, device_map):
        print('Loading model from...', model_root_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_root_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = RobertaForSequenceClassification.from_pretrained(model_root_path, num_labels = len(labels), device_map=device_map)
                                                 
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.id2label = id2label
        model.config.label2id = label2id

        return model, tokenizer