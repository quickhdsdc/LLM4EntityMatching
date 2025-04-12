import ast
import random


class DataPreprocessor:
    def __init__(self) -> None:      
        print('Preprocessing the data...')

    def preprocess_data(self, dataset, tokenizer, input_type='text_on', max_length=256, num_neg=9, remove_unused_col=True, is_train:bool=True):

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

        sample['text'] = text_query
        sample['candidates'] = shuffled_candidates   
        sample['labels'] = shuffled_labels

        queries_encodings = self.tokenizer(sample['text'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_query = queries_encodings["input_ids"]
        attention_mask_query = queries_encodings["attention_mask"]

        candidates_encodings = self.tokenizer(list(sample['candidates']), max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_candidates = candidates_encodings["input_ids"]
        attention_mask_candidates = candidates_encodings["attention_mask"]

        sample['input_ids'] = input_ids_query
        sample['attention_mask'] = attention_mask_query
        sample['input_ids_query'] = input_ids_query
        sample['attention_mask_query'] = attention_mask_query
        sample['input_ids_candidates'] = input_ids_candidates
        sample['attention_mask_candidates'] = attention_mask_candidates

        return sample


class DataPreprocessor_pairwise:
    def __init__(self) -> None:
        '''
        '''