from pathlib import Path
from functools import partial
import random


class DataPreprocessor_pairwise:
    def __init__(self) -> None:
        '''
        '''

        print('Preprocessing the data...ft_roberta_c')

    def preprocess_data(self, dataset, label2id, tokenizer, args=None, remove_unused_col=True, is_train=True):
        """
        :param model: Hugging Face model
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param dataset (str): Instruction dataset
        """
        # perpare input text and label
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.input_type = args.input_type
        self.task_name = args.task_name
        remove_cols = [n for n in dataset.column_names if n not in ["text", "label", "input_ids", "attention_mask"]]
        if remove_unused_col:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x,self.input_type,self.task_name),
                                  remove_columns=remove_cols, keep_in_memory=True)
        else:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x,self.input_type,self.task_name), keep_in_memory=True)
        seed = 42
        if is_train==True:
            dataset = dataset.shuffle(seed = seed)
        return dataset

    def create_input_text_and_label(self, sample, input_type, task_name):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction = "You will read two sentence-like entities to be matched. Each entity has several attributes." \
                      "Your task is to decide whether the two entities are matched (they refer to the same entity). if matched, please answer 'yes'. If not, answer 'no'."
        if 'Amazon_Google' in task_name or 'Amazon-Google' in task_name:
            text_src = 'title: ' + sample['title_left'] + '. manufacturer: ' + sample['manufacturer_left'] + '. price: ' + \
                    str(sample['price_left'])
            text_tgt = 'title: ' + sample['title_right'] + '. manufacturer: ' + sample['manufacturer_right'] + '. price: ' + \
                    str(sample['price_right'])
        elif 'ECLASS' in task_name:
            text_src = 'name: ' + sample['idShort_left'] + '. description: ' + sample['description_left']
            text_tgt = 'name: ' + sample['name_right'] + '. definition: ' + sample['definition_right']
        elif 'Manufactur' in task_name:
            text_src = 'name: ' + sample['Name_left'] + '. value: ' + sample['Value_left'] + '. Unit: ' + sample['Units_left']
            text_tgt = 'name: ' + sample['idShort_right'] + '. description: ' + sample['description_right'] 
        elif 'Abt' in task_name:
            text_src = 'name: ' + sample['name_left'] + '. description: ' + sample['description_left'] + '. price: ' + \
                    sample['price_left']
            text_tgt = 'name: ' + sample['name_right'] + '. description: ' + sample['description_right'] + '. price: ' + \
                    sample['price_right']         
        elif 'Walmart' in task_name:
            text_src = 'title: ' + sample['title_left'] + '. category: ' + sample['category_left'] + '. price: ' + \
                   str(sample['price_left']) + '. modelno: ' + sample['modelno_left'] + '. brand: ' + sample['brand_left']
            text_tgt = 'title: ' + sample['title_right'] + '. category: ' + sample['category_right'] + '. price: ' + \
                   str(sample['price_right']) + '. modelno: ' + sample['modelno_right'] + '. brand: ' + sample['brand_right']   
        elif 'DBLP' in task_name:
            text_src = 'title: ' + sample['title_left'] + '. authors: ' + sample['authors_left'] + '. venue: ' + \
                        str(sample['venue_left']) + '. year: ' + str(sample['year_left'])
            text_tgt = 'title: ' + sample['title_right'] + '. authors: ' + sample['authors_right'] + '. venue: ' + \
                        str(sample['venue_right']) + '. year: ' + str(sample['year_right'])

        if input_type == 'text_st_on':
            sample["text"] = f"<s>{text_src}</s><s>{text_tgt}</s>"
        elif input_type == 'text_on':
            sample["text"] = f"[CLS]{text_src}[SEP]{text_tgt}[SEP]"
        elif input_type == 'inst_text_st_on':
            sample["text"] = instruction + '<s>' + text_src + '</s><s>' + text_tgt + '</s>'
        elif input_type == 'inst_text_on':
            sample["text"] = instruction + '\n' + text_src + '\n' + text_tgt

        if isinstance(sample['label'], str):
            sample['label'] = self.label2id[sample['label']]

        sample['input_ids'] = \
        self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True,
                                   return_tensors="pt")["input_ids"]
        sample['attention_mask'] = \
        self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True,
                                   return_tensors="pt")["attention_mask"]
        return sample
    

class DataPreprocessor:
    def __init__(self) -> None:
        '''
        '''