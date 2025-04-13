from pathlib import Path
from functools import partial
import random

# Initialize static strings for the prompt template
INSTRUCTION_KEY = "### Instruction:" #"<instruction>" 
INSTRUCTION_KEY_END = ''
INPUT_KEY = "INPUT:" 
INPUT_KEY_END = ''
ENTITY1 = 'ENTITY1:'
ENTITY1_END = ''
ENTITY2 = 'ENTITY2:'
ENTITY2_END = ''
RESPONSE_KEY = 'RESPONSE:'
END_KEY = '### End'

INSTRUCTION_KEY_ST = "<instruction>" 
INSTRUCTION_KEY_END_ST = '</instruction>'
INPUT_KEY_ST = '<input>' #"EDIT:" 
INPUT_KEY_END_ST = '</input>'
ENTITY1_ST = '<entity1>'
ENTITY1_END_ST = '</entity1>'
ENTITY2_ST = '<entity2>'
ENTITY2_END_ST = '</entity2>'
RESPONSE_KEY_ST = "<response>" #'LABEL:'
END_KEY_ST = "</response>" #'### End'


TASK_PROMPT0 = "You will read two sentence-like entities to be matched. Each entity has several attributes such as name and description.Your task is to decide whether the two entities are matched (they refer to the same entity). Only answer 'yes' or 'no'."
COT_PROMPT =  "Think step by step. First, entities may be professional terminologies in specific domains, you should consider the domain knowledge. Second, the entity names and descriptions or definitions are most important. should consider synonyms. Third, entity1 may be defined for a specific use case or domain, while entity2 may be defined in more general terms. If the scope of Entity1 belongs to that of Entity2, they should be considered matching. However, if the scope of Entity2 belongs to that of Entity1, they should be considered not matching. Below are several examples"
# there could more types of prompts defined in PROMPT_DIC
PROMPT_DIC ={'pt0':TASK_PROMPT0}
#others: ep1_pt0, ep1_pt1, ep-all_pt0, ep-all_pt1, ep-random1_pt0, ep-random1_pt1
PROMPT_ST_DIC = {'nl': [INSTRUCTION_KEY,INSTRUCTION_KEY_END, INPUT_KEY, INPUT_KEY_END, ENTITY1, ENTITY1_END, ENTITY2, ENTITY2_END, RESPONSE_KEY, END_KEY],
                 'st': [INSTRUCTION_KEY_ST,INSTRUCTION_KEY_END_ST, INPUT_KEY_ST, INPUT_KEY_END_ST, ENTITY1_ST, ENTITY1_END_ST, ENTITY2_ST, ENTITY2_END_ST, RESPONSE_KEY_ST, END_KEY_ST]}

class DataPreprocessor_pairwise:
    def __init__(self) -> None:
        '''
        '''
        
        print('Preprocessing the data...')

    def preprocess_data(self, tokenizer, dataset, args=None, is_train:bool=True, prompt_type:str='pt0', prompt_st_type:str='nl'):
        """
        :param model: Hugging Face model
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param dataset (str): Instruction dataset
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.prompt_st_type = prompt_st_type
        self.task_name = args.task_name
        self.max_length = args.max_length
        instruction_key, instruction_key_end, input_key, input_key_end, old_start, entity1_end, entity2, entity2_end, response_key, end_key = PROMPT_ST_DIC[self.prompt_st_type]
        seed = 42
        # Add prompt to each sample
        print("Preprocessing dataset...")
        remove_cols = [n for n in dataset.column_names if n not in ["text","label", "input_ids_text", "attention_mask_text"]]
        if is_train:
            dataset = dataset.map(self.create_prompt_formats_train, remove_columns = remove_cols, keep_in_memory=True)
        else:
            dataset = dataset.map(self.create_prompt_formats_test, keep_in_memory=True)

        # Shuffle dataset
        if is_train:
            dataset = dataset.shuffle(seed = seed)
        return dataset, response_key
    
    def preprocess_batch(self, batch, tokenizer, max_length):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenized = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors=None
        )
        input_ids = tokenized["input_ids"]
        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids]
            for ids in input_ids
        ]
        tokenized["labels"] = labels
        return tokenized
    
    def create_task_prompt(self, prompt_type):
        """
        Creates a task prompt based on the prompt type
        :param prompt_type: Type of the prompt
        """
        if prompt_type == 'pt0':
            p = TASK_PROMPT0
        elif prompt_type == 'pt1':
            EXAMPLES = self.create_examples(3, self.dataset)
            p = TASK_PROMPT0 + "\n" + EXAMPLES
        elif prompt_type == 'pt2':
            EXAMPLES = self.create_examples(3, self.dataset)
            p = TASK_PROMPT0 + "\n" + EXAMPLES+ "\n" + COT_PROMPT
        else:
            assert False, f"Invalid prompt type: {prompt_type}"
        return p


    def create_examples(self, n_sample, dataset):
        """
        Creates examples for ICL and CoT
        """
        pos_ds = dataset.filter(lambda example: example['label'] == 'yes')
        neg_ds = dataset.filter(lambda example: example['label'] == 'no')
        if not pos_ds or not neg_ds:
            pos_ds = dataset.filter(lambda example: example['label'] == 1)
            neg_ds = dataset.filter(lambda example: example['label'] == 0)
        pos_samples = pos_ds.shuffle(seed=42).select(range(n_sample))
        neg_samples = neg_ds.shuffle(seed=42).select(range(n_sample))

        def add_examples(prefix, examples, naming='Entity'):
            for row in examples:
                if 'Amazon_Google' in self.task_name or 'Amazon-Google' in self.task_name:
                    entity1 = 'title: ' + row['title_left'] + '. manufacturer: ' + row['manufacturer_left'] + '. price: ' + str(row['price_left'])
                    entity2 = 'title: ' + row['title_right'] + '. manufacturer: ' + row['manufacturer_right'] + '. price: ' + str(row['price_right'])

                elif 'ECLASS' in self.task_name:
                    entity1 = 'name: ' + row['idShort_left'] + '. description: ' + row['description_left']
                    entity2 = 'name: ' + row['name_right'] + '. definition: ' + row['definition_right']

                elif 'Manufactur' in self.task_name:
                    entity1 = 'name: ' + row['Name_left'] + '. value: ' + row['Value_left'] + '. Unit: ' + row['Units_left']
                    entity2 = 'name: ' + row['idShort_right'] + '. description: ' + row['description_right']

                elif 'Abt' in self.task_name:
                    entity1 = 'name: ' + row['name_left'] + '. description: ' + row['description_left'] + '. price: ' + row['price_left']
                    entity2 = 'name: ' + row['name_right'] + '. description: ' + row['description_right'] + '. price: ' + row['price_right']

                elif 'Walmart' in self.task_name:
                    entity1 = 'title: ' + row['title_left'] + '. category: ' + row['category_left'] + '. price: ' + str(row['price_left']) + '. modelno: ' + row['modelno_left'] + '. brand: ' + row['brand_left']
                    entity2 = 'title: ' + row['title_right'] + '. category: ' + row['category_right'] + '. price: ' + str(row['price_right']) + '. modelno: ' + row['modelno_right'] + '. brand: ' + row['brand_right']

                elif 'DBLP' in self.task_name:
                    entity1 = 'title: ' + row['title_left'] + '. authors: ' + row['authors_left'] + '. venue: ' + str(row['venue_left']) + '. year: ' + str(row['year_left'])
                    entity2 = 'title: ' + row['title_right'] + '. authors: ' + row['authors_right'] + '. venue: ' + str(row['venue_right']) + '. year: ' + str(row['year_right'])

                prefix = f"{prefix}{naming}1: '{entity1}' {naming}2: '{entity2}'\n"
            return prefix

        match_prefix = "Matches:\n"
        non_match_prefix = "Non-matches:\n"

        match_prefix = add_examples(match_prefix, pos_samples)
        non_match_prefix = add_examples(non_match_prefix, neg_samples)

        final_input = f"Below are several examples:\n{match_prefix}\n{non_match_prefix}"

        return final_input


        
    def create_prompt_formats_train(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction_key, instruction_key_end, input_key, input_key_end, entity1, entity1_end, entity2, entity2_end, response_key, end_key = PROMPT_ST_DIC[self.prompt_st_type]
        task_prompt = self.create_task_prompt(self.prompt_type)
        # Combine a prompt with the static strings
        instruction = f"{instruction_key} {task_prompt} {instruction_key_end}"

        if 'Amazon_Google' in self.task_name or 'Amazon-Google' in self.task_name:
            text_src = 'title: ' + sample['title_left'] + '. manufacturer: ' + sample['manufacturer_left'] + '. price: ' + str(sample['price_left'])
            text_tgt = 'title: ' + sample['title_right'] + '. manufacturer: ' + sample['manufacturer_right'] + '. price: ' + str(sample['price_right'])

        elif 'ECLASS' in self.task_name:
            text_src = 'name: ' + sample['idShort_left'] + '. description: ' + sample['description_left']
            text_tgt = 'name: ' + sample['name_right'] + '. definition: ' + sample['definition_right']

        elif 'Manufactur' in self.task_name:
            text_src = 'name: ' + sample['Name_left'] + '. value: ' + sample['Value_left'] + '. Unit: ' + sample['Units_left']
            text_tgt = 'name: ' + sample['idShort_right'] + '. description: ' + sample['description_right']

        elif 'Abt' in self.task_name:
            text_src = 'name: ' + sample['name_left'] + '. description: ' + sample['description_left'] + '. price: ' + sample['price_left']
            text_tgt = 'name: ' + sample['name_right'] + '. description: ' + sample['description_right'] + '. price: ' + sample['price_right']

        elif 'Walmart' in self.task_name:
            text_src = 'title: ' + sample['title_left'] + '. category: ' + sample['category_left'] + '. price: ' + str(sample['price_left']) + '. modelno: ' + sample['modelno_left'] + '. brand: ' + sample['brand_left']
            text_tgt = 'title: ' + sample['title_right'] + '. category: ' + sample['category_right'] + '. price: ' + str(sample['price_right']) + '. modelno: ' + sample['modelno_right'] + '. brand: ' + sample['brand_right']

        elif 'DBLP' in self.task_name:
            text_src = 'title: ' + sample['title_left'] + '. authors: ' + sample['authors_left'] + '. venue: ' + str(sample['venue_left']) + '. year: ' + str(sample['year_left'])
            text_tgt = 'title: ' + sample['title_right'] + '. authors: ' + sample['authors_right'] + '. venue: ' + str(sample['venue_right']) + '. year: ' + str(sample['year_right'])

        if sample['label'] == 1:
            label = 'yes'
        elif sample['label'] == 0:
            label = 'no'
        else:
            label = sample['label'] 
        input_context = f"{input_key}\n {entity1} {text_src} {entity1_end}\n {entity2} {text_tgt} {entity2_end}\n{input_key_end}"
        response = f"{response_key}{label}"
        end = f"{end_key}"
        # Create a list of prompt template elements
        parts = [part for part in [instruction, input_context, response, end] if part]
        # Join prompt template elements into a single string to create the prompt template
        formatted_prompt = "\n".join(parts)
        # Store the formatted prompt template in a new key "text"
        sample["text"] = formatted_prompt

        text_encodings = self.tokenizer(sample['text'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_text = text_encodings["input_ids"]
        attention_mask_text = text_encodings["attention_mask"]
        sample['input_ids_text'] = input_ids_text
        sample['attention_mask_text'] = attention_mask_text

        # label_encodings = self.tokenizer(sample['label'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        # input_ids_label = label_encodings["input_ids"]
        # attention_mask_label = label_encodings["attention_mask"]
        # sample['input_ids_label'] = input_ids_label
        # sample['attention_mask_label'] = attention_mask_label

        if sample['label'] == 'yes':
            sample['label'] = 1
        elif sample['label'] == 'no':
            sample['label'] = 0

        return sample
    
    def create_prompt_formats_test(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction_key, instruction_key_end, input_key, input_key_end, entity1, entity1_end, entity2, entity2_end, response_key, end_key = PROMPT_ST_DIC[self.prompt_st_type]
        task_prompt = self.create_task_prompt(self.prompt_type)
        instruction = f"{instruction_key} {task_prompt} {instruction_key_end}"
        if 'Amazon_Google' in self.task_name or 'Amazon-Google' in self.task_name:
            text_src = 'title: ' + sample['title_left'] + '. manufacturer: ' + sample['manufacturer_left'] + '. price: ' + str(sample['price_left'])
            text_tgt = 'title: ' + sample['title_right'] + '. manufacturer: ' + sample['manufacturer_right'] + '. price: ' + str(sample['price_right'])

        elif 'ECLASS' in self.task_name:
            text_src = 'name: ' + sample['idShort_left'] + '. description: ' + sample['description_left']
            text_tgt = 'name: ' + sample['name_right'] + '. definition: ' + sample['definition_right']

        elif 'Manufactur' in self.task_name:
            text_src = 'name: ' + sample['Name_left'] + '. value: ' + sample['Value_left'] + '. Unit: ' + sample['Units_left']
            text_tgt = 'name: ' + sample['idShort_right'] + '. description: ' + sample['description_right']

        elif 'Abt' in self.task_name:
            text_src = 'name: ' + sample['name_left'] + '. description: ' + sample['description_left'] + '. price: ' + sample['price_left']
            text_tgt = 'name: ' + sample['name_right'] + '. description: ' + sample['description_right'] + '. price: ' + sample['price_right']

        elif 'Walmart' in self.task_name:
            text_src = 'title: ' + sample['title_left'] + '. category: ' + sample['category_left'] + '. price: ' + str(sample['price_left']) + '. modelno: ' + sample['modelno_left'] + '. brand: ' + sample['brand_left']
            text_tgt = 'title: ' + sample['title_right'] + '. category: ' + sample['category_right'] + '. price: ' + str(sample['price_right']) + '. modelno: ' + sample['modelno_right'] + '. brand: ' + sample['brand_right']

        elif 'DBLP' in self.task_name:
            text_src = 'title: ' + sample['title_left'] + '. authors: ' + sample['authors_left'] + '. venue: ' + str(sample['venue_left']) + '. year: ' + str(sample['year_left'])
            text_tgt = 'title: ' + sample['title_right'] + '. authors: ' + sample['authors_right'] + '. venue: ' + str(sample['venue_right']) + '. year: ' + str(sample['year_right'])

        input_context = f"{input_key}\n {entity1} {text_src} {entity1_end}\n {entity2} {text_tgt} {entity2_end}\n{input_key_end}"
        response = f"{response_key}"
        parts = [part for part in [instruction, input_context, response] if part]
        formatted_prompt = "\n".join(parts)
        sample["text"] = formatted_prompt

        text_encodings = self.tokenizer(sample['text'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_text = text_encodings["input_ids"]
        attention_mask_text = text_encodings["attention_mask"]
        sample['input_ids_text'] = input_ids_text
        sample['attention_mask_text'] = attention_mask_text

        # label_encodings = self.tokenizer(sample['label'], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        # input_ids_label = label_encodings["input_ids"]
        # attention_mask_label = label_encodings["attention_mask"]
        # sample['input_ids_label'] = input_ids_label
        # sample['attention_mask_label'] = attention_mask_label

        if sample['label'] == 'yes':
            sample['label'] = 1
        elif sample['label'] == 'no':
            sample['label'] = 0
        return sample
        
class DataPreprocessor:
    def __init__(self) -> None:
        '''
        '''