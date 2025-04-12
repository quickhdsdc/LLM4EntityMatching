import ast
import random

class DataPreprocessor_pairwise:
    def __init__(self) -> None:
        '''
        '''
        
        print('Preprocessing the data...ft_llm_c')

    def preprocess_data(self, dataset, label2id, input_type='text_st_on', has_label=True, remove_unused_col=False, is_train:bool=True):
        """
        :param model: Hugging Face model
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param dataset (str): Instruction dataset
        """
        # perpare input text and label
        self.label2id = label2id
        self.input_type = input_type
        self.has_label = has_label
        self.is_train = is_train
        remove_cols = [n for n in dataset.column_names if n not in ["text","label"]]
        if remove_unused_col:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type), remove_columns = remove_cols, keep_in_memory=True)
        else:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type), keep_in_memory=True)
        if self.is_train:
            seed = 42
            dataset = dataset.shuffle(seed = seed)
        return dataset
    
    def create_input_text_and_label(self, sample, input_type):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction = "You will read two sentence-like entities to be matched. Each entity has several attributes."\
                      "Your task is to decide whether the two entities are matched (they refer to the same entity). if matched, please answer 'yes'. If not, answer 'no'." 
        text_src = 'name: ' + sample['idShort_left'] + '. description: ' + sample['description_left']
        text_tgt = 'name: ' + sample['name_right'] + '. definition: ' + sample['definition_right']

        if input_type == 'text_st_on':
            sample["text"] = f"<entity1> {text_tgt} </entity1>" + '\n ' + f"<entity2> {text_src} </entity2>"
        elif input_type == 'text_on':
            sample["text"] = text_tgt + '\n ' + text_src
        elif input_type == 'inst_text_st_on':
            sample["text"] = instruction + '\n ' + f"<entity1> {text_tgt} </entity1>" + '\n ' + f"<entity2> {text_src} </entity2>"
        elif input_type == 'inst_text_on':
            sample["text"] = instruction + '\n ' + text_tgt + '\n ' + text_src
            
        if self.has_label:
            label = self.label2id[sample['label']]
            sample["label"] = label

        return sample


class DataPreprocessor:
    def __init__(self) -> None:
        '''
        '''

        print('Preprocessing the data...')

    def preprocess_data(self, dataset, input_type='text_on', max_length=256, num_neg=9, remove_unused_col=True, is_train: bool = True):
        """
        :param model: Hugging Face model
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param dataset (str): Instruction dataset
        """
        self.num_neg = num_neg
        self.max_length = max_length
        self.num_neg = num_neg
        self.input_type = input_type
        self.is_train = is_train
        # perpare input text and label
        remove_cols = [n for n in dataset.column_names if
                       n not in ["text", "labels", "candidates", "text_src"]]
        if remove_unused_col:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x), remove_columns=remove_cols)
        else:
            dataset = dataset.map(lambda x: self.create_input_text_and_label(x))
        if self.is_train:
            # Shuffle dataset
            seed = 42
            dataset = dataset.shuffle(seed=seed)
        return dataset

    def create_input_text_and_label(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        :param n_neg: number of negative samples to include
        """
        instruction = "Given a query entity consisting of several attributes, select the most similar entity that semantically matches the query entity from the given candidates. only return the index of the selected one without any explanations. For example, when the answer is index '3', which refers to candidate4 which is the most matched candidate for the given query entity."
        text_query = sample['text_src']
        labels = ast.literal_eval(sample['labels'])

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

        return sample