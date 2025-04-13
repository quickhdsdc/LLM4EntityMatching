from pathlib import Path
from functools import partial
import random
import ast


class DataPreprocessor:
    def __init__(self) -> None:
        '''
        '''
        
        print('Preprocessing the data...')

    def preprocess_data(self, dataset, tokenizer, input_type='text_on', max_length=256, num_neg=9, remove_unused_col=True, is_train:bool=True):
        """
        :param model: Hugging Face model
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param dataset (str): Instruction dataset
        """
        self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_neg = num_neg
        self.input_type = input_type
        self.is_train = is_train
        # perpare input text and label
        remove_cols = [n for n in dataset.column_names if n not in ["text","input_ids_query","attention_mask_query","input_ids_instr","attention_mask_instr","input_ids_candidates","attention_mask_candidates","labels","candidates","text_src"]]
        if remove_unused_col:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x), remove_columns = remove_cols, keep_in_memory=True)
        else:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x), keep_in_memory=True)
        if self.is_train:
            # Shuffle dataset
            seed = 42
            dataset = dataset.shuffle(seed = seed)
        return dataset

    def create_input_text_and_label(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        :param n_neg: number of negative samples to include
        """
        instruction = "Given a search query of an entity consisting of several attributes, retrieve a similar entity that semantically matches the query entity from the given candidates. /n"
        text_query = sample['text_src']
        labels = ast.literal_eval(sample['labels'])

        text_candidates = sample['candidates']

        # Pair candidates with their corresponding labels
        paired_candidates = list(zip(text_candidates, labels))

        # Separate positive and negative samples
        positive_samples = [pair for pair in paired_candidates if pair[1] == 1]
        negative_samples = [pair for pair in paired_candidates if pair[1] == 0]

        if self.is_train:
            # Select n_neg negative samples
            selected_negative_samples = random.sample(negative_samples, min(self.num_neg, len(negative_samples)))
            # Combine the positive sample with selected negative samples
            selected_samples = positive_samples + selected_negative_samples
            # Shuffle the selected samples
            random.shuffle(selected_samples)
            # Unzip the selected samples back into separate lists
            shuffled_candidates, shuffled_labels = zip(*selected_samples)
        else:
            shuffled_candidates = text_candidates
            shuffled_labels = labels

        if self.input_type == 'text_on':
            sample['text'] = text_query
        elif self.input_type == 'inst_text_on':
            sample['text'] = instruction + ' ' + text_query
    
        sample['instr'] = instruction
        sample['candidates'] = shuffled_candidates   
        sample['labels'] = shuffled_labels

        queries_encodings = self.tokenizer(sample['text'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_query = queries_encodings["input_ids"]
        attention_mask_query = queries_encodings["attention_mask"]

        instr_encodings = self.tokenizer(sample['instr'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_instr = instr_encodings["input_ids"]
        attention_mask_instr = instr_encodings["attention_mask"]

        candidates_encodings = self.tokenizer(list(sample['candidates']), max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_candidates = candidates_encodings["input_ids"]
        attention_mask_candidates = candidates_encodings["attention_mask"]

        sample['input_ids_query'] = input_ids_query
        sample['attention_mask_query'] = attention_mask_query
        sample['input_ids_instr'] = input_ids_instr
        sample['attention_mask_instr'] = attention_mask_instr
        sample['input_ids_candidates'] = input_ids_candidates
        sample['attention_mask_candidates'] = attention_mask_candidates

        return sample
    

class DataPreprocessor_pairwise:
    def __init__(self) -> None:
        '''
        '''
        
        print('Preprocessing the data...ft_llm_c')

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
        print("Preprocessing dataset...ft_llm_c")
        remove_cols = [n for n in dataset.column_names if n not in ["text","label", "input_ids_text", "attention_mask_text"]]
        if remove_unused_col:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type, self.task_name), remove_columns = remove_cols, keep_in_memory=True)
        else:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type, self.task_name), keep_in_memory=True)
        seed = 42
        if is_train==True:
            dataset = dataset.shuffle(seed = seed)
        max_len = 0
        for example in dataset:
            max_len = max(max_len, sum(example['attention_mask_text'][0]))
        print(f"\nmax_len: {max_len}\n")
        return dataset
    
    def create_input_text_and_label(self, sample, input_type, task_name):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction = "You will read two sentence-like entities to be matched. Each entity has several attributes."\
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
            sample["text"] = f"<entity1> {text_src} </entity1>" + '\n ' + f"<entity2> {text_tgt} </entity2>"
        elif input_type == 'text_on':
            sample["text"] = text_src + '\n ' + text_tgt
        elif input_type == 'inst_text_st_on':
            sample["text"] = instruction + '\n ' + f"<entity1> {text_src} </entity1>" + '\n ' + f"<entity2> {text_tgt} </entity2>"
        elif input_type == 'inst_text_on':
            sample["text"] = instruction + '\n ' + text_src + '\n ' + text_tgt
            
        if isinstance(sample['label'], str):
            sample['label'] = self.label2id[sample['label']]

        sample['input_ids_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        sample['attention_mask_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
        return sample
    
        

def main():
    ''

if __name__ == "__main__":
    main()