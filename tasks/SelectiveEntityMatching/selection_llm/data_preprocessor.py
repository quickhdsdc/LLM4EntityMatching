import ast
import random


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

        if isinstance(sample['candidates'], str):
            try:
                text_candidates = ast.literal_eval(sample['candidates'])
            except ValueError:
                raise ValueError("The 'candidates' field cannot be parsed into a list.")

        # If the candidates are already a list, assign them directly
        else:
            text_candidates = sample['candidates']

        # Pair candidates with their corresponding labels
        paired_candidates = list(zip(text_candidates, labels))

        # Separate positive and negative samples
        positive_samples = [pair for pair in paired_candidates if pair[1] == 1]
        negative_samples = [pair for pair in paired_candidates if pair[1] == 0]

        if self.is_train:
            selected_negative_samples = random.sample(negative_samples, min(self.num_neg, len(negative_samples)))
            selected_samples = positive_samples + selected_negative_samples
            random.shuffle(selected_samples)
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

    def preprocess_data(self, dataset, label2id, tokenizer, max_length=1024, input_type='text_st_on', has_label=True, remove_unused_col=False, is_train:bool=True):
        """
        :param model: Hugging Face model
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param dataset (str): Instruction dataset
        """
        # perpare input text and label
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_type = input_type
        self.has_label = has_label
        print("Preprocessing dataset...ft_llm_c")
        print("input_type", input_type)
        print('max_length', max_length)
        remove_cols = [n for n in dataset.column_names if n not in ["text","label", "input_ids_text", "attention_mask_text"]]
        print(remove_cols)
        if remove_unused_col:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type), remove_columns = remove_cols, keep_in_memory=True)
        else:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type), keep_in_memory=True)
        if is_train:
            seed = 42
            dataset = dataset.shuffle(seed = seed)
        max_len = 0
        for example in dataset:
            max_len = max(max_len, sum(example['attention_mask_text'][0]))
        print(f"\nmax_len: {max_len}\n")
        return dataset
    
    def create_input_text_and_label(self, sample, input_type):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction = "You will read two sentence-like entities to be matched. Each entity has several attributes."\
                      "Your task is to decide whether the two entities are matched (they refer to the same entity). if matched, please answer 'yes'. If not, answer 'no'." 
        # text_src = 'title: ' + sample['title_left'] + '. manufacturer: ' + sample['manufacturer_left'] + '. price: ' + \
        #            str(sample['price_left'])
        # text_tgt = 'title: ' + sample['title_right'] + '. manufacturer: ' + sample['manufacturer_right'] + '. price: ' + \
        #            str(sample['price_right'])

        text_src = sample['title_left'] + '. ' + sample['manufacturer_left'] + '. ' + str(sample['price_left'])
        text_tgt = sample['title_right'] + '. ' + sample['manufacturer_right'] + '. ' + str(sample['price_right'])


        if input_type == 'text_st_on':
            sample["text"] = f"<entity1> {text_src} </entity1>" + '\n ' + f"<entity2> {text_tgt} </entity2>"
        elif input_type == 'text_on':
            sample["text"] = text_src + '\n ' + text_tgt
        elif input_type == 'inst_text_st_on':
            sample["text"] = instruction + '\n ' + f"<entity1> {text_src} </entity1>" + '\n ' + f"<entity2> {text_tgt} </entity2>"
        elif input_type == 'inst_text_on':
            sample["text"] = instruction + '\n ' + text_src + '\n ' + text_tgt
            
        if self.has_label:
            label = self.label2id[sample['label']]
            sample["label"] = label

        sample["text_src"] = text_src
        sample["text_tgt"] = text_tgt

        sample['input_ids_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        sample['attention_mask_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
        sample['input_ids_query'] = self.tokenizer.encode_plus(sample["text_src"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        sample['attention_mask_query'] = self.tokenizer.encode_plus(sample["text_src"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
        sample['input_ids_candidate'] = self.tokenizer.encode_plus(sample["text_tgt"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        sample['attention_mask_candidate'] = self.tokenizer.encode_plus(sample["text_tgt"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
        return sample