from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from data_representation import DeepMatcherProcessor
class TaskDataLoader:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str=None, imbalance_ratio:int=1) -> None:

        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/_entity matching') / task_name
        processor = DeepMatcherProcessor()
        train_df = processor.get_train_examples(self.task_data_dir)
        
        # balance the training data with ratio 
        pos_df = train_df[train_df['label'] == 'yes']
        neg_df = train_df[train_df['label'] == 'no']
        max_neg_samples = len(pos_df) * imbalance_ratio

        if len(neg_df) > max_neg_samples:
            neg_df_downsampled = neg_df.sample(n=max_neg_samples, random_state=42)
        else:
            neg_df_downsampled = neg_df
        train_df_balanced = pd.concat([pos_df, neg_df_downsampled])
        train_df_balanced = train_df_balanced.sample(frac=1).reset_index(drop=True)        

        train_df_balanced.to_csv(self.task_data_dir/'train_df.csv', index=False)
        valid_df = processor.get_valid_examples(self.task_data_dir)
        valid_df.to_csv(self.task_data_dir/'valid_df.csv', index=False)
        test_df = processor.get_test_examples(self.task_data_dir)
        test_df.to_csv(self.task_data_dir/'test_df.csv', index=False)

        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('df.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False, cache_dir=self.task_data_dir)

        label_names = sorted(set(label for label in self.dataset["train_df"]["label"]))
        self.labels = label_names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = i
            id2label[i] = label
        self.label2id = label2id
        self.id2label = id2label

    def load_train(self):
        return self.dataset[self.train_type]
    def load_val(self):
        return self.dataset[self.val_type]
    def load_test(self):
        if self.test_type is None:
            return None
        return self.dataset[self.test_type]
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()
    def get_labels(self):
        return self.labels, self.label2id, self.id2label


class TaskDataLoader_pairwise:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str=None, imbalance_ratio:int=1) -> None:

        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/_entity matching') / task_name 

        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('df_pairwise.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False, cache_dir=self.task_data_dir)

        label_names = sorted(set(label for label in self.dataset["train_df_pairwise"]["label"]))
        self.labels = label_names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = i
            id2label[i] = label
        self.label2id = label2id
        self.id2label = id2label

    def load_train(self):
        return self.dataset[self.train_type]
    def load_val(self):
        return self.dataset[self.val_type]
    def load_test(self):
        if self.test_type is None:
            return None
        return self.dataset[self.test_type]
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()
    def get_labels(self):
        return self.labels, self.label2id, self.id2label


class TaskDataLoader_retrieval:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str, task_name_test = None, no_val:bool=False) -> None:

        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/_entity matching') / task_name
        self.no_val = no_val
        self.task_name_test = task_name_test

        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('df_new.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False, cache_dir=self.task_data_dir)
        if task_name_test != None:
            data_files_test = str(Path('data/_entity matching') / task_name_test / 'test_df_new.csv')
            self.dataset_test = load_dataset("csv", data_files = data_files_test, keep_default_na=False, cache_dir=self.task_data_dir)
        
    def load_train(self):
        if self.no_val:
            train_ds = concatenate_datasets([self.dataset[self.train_type], self.dataset[self.val_type]])
        else:
            train_ds = self.dataset[self.train_type]
        train_ds = train_ds.shuffle(seed=42)
        return train_ds
    def load_val(self):
        val_ds = self.dataset[self.val_type]
        val_ds = val_ds.shuffle(seed=42)
        return val_ds
    def load_test(self):
        if self.task_name_test == None:
            test_ds = self.dataset[self.test_type]
        else:
            test_ds = self.dataset_test['train']
        return test_ds
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()



class TaskDataLoader_aas:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str=None, imbalance_ratio:int=1, no_val:bool=False) -> None:

        self.imbalance_ratio = imbalance_ratio
        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/_entity matching') / task_name
        self.no_val = no_val

        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('df.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False, cache_dir=self.task_data_dir)

        label_names = sorted(set(label for label in self.dataset["train_df"]["label"]))
        self.labels = label_names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = i
            id2label[i] = label
        self.label2id = label2id
        self.id2label = id2label

    def load_train(self):
        if self.no_val:
            train_ds = concatenate_datasets([self.dataset[self.train_type], self.dataset[self.val_type]])
        else:
            train_ds = self.dataset[self.train_type]
        pos_ds = train_ds.filter(lambda example: example['label'] == 'yes')
        neg_ds = train_ds.filter(lambda example: example['label'] == 'no')
        # imbalance ratio
        max_neg_samples = self.imbalance_ratio * len(pos_ds)

        if len(neg_ds) > max_neg_samples:
            neg_ds = neg_ds.shuffle(seed=42).select(range(max_neg_samples))

        train_ds_balanced = concatenate_datasets([pos_ds, neg_ds])
        train_ds_balanced = train_ds_balanced.shuffle(seed=42)

        return train_ds_balanced
    
    def load_val(self):
        val_ds = self.dataset[self.val_type]
        val_ds = val_ds.shuffle(seed=42)
        return val_ds
    def load_test(self):
        if self.test_type is None:
            return None
        else:
            test_ds = self.dataset[self.test_type]
        return test_ds
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()
    def get_labels(self):
        return self.labels, self.label2id, self.id2label
        

class TaskDataLoader_aas_retrieval:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str=None, no_val:bool=True) -> None:

        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/_entity matching') / task_name
        self.no_val = no_val

        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('df.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False, cache_dir=self.task_data_dir)
        
    def load_train(self):
        if self.no_val:
            train_ds = concatenate_datasets([self.dataset[self.train_type], self.dataset[self.val_type]])
        else:
            train_ds = self.dataset[self.train_type]
        train_ds = train_ds.shuffle(seed=42)
        return train_ds
    def load_val(self):
        val_ds = self.dataset[self.val_type]
        val_ds = val_ds.shuffle(seed=42)
        return val_ds
    def load_test(self):
        if self.test_type is None:
            return None
        else:
            test_ds = self.dataset[self.test_type]
        return test_ds
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()


class TaskDataLoader_aas_retrieval_testOnly:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str=None) -> None:

        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/_entity matching') / task_name

        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('df.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False, cache_dir=self.task_data_dir)

        label_names = sorted(set(label for label in self.dataset["train_df"]["label"]))
        self.labels = label_names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = i
            id2label[i] = label
        self.label2id = label2id
        self.id2label = id2label
        
    def load_train(self):
        train_ds = self.dataset[self.train_type]
        train_ds = train_ds.shuffle()
        return train_ds
    def load_val(self):
        val_ds = self.dataset[self.val_type]
        val_ds = val_ds.shuffle()
        return val_ds
    def load_test(self):
        if self.test_type is None:
            return None
        else:
            # test_ds = concatenate_datasets([self.dataset[self.train_type], self.dataset[self.val_type], self.dataset[self.test_type]])
            test_ds = self.dataset[self.test_type]
            # test_ds = test_ds.shuffle(seed=42)
        return test_ds
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()
    def get_labels(self):
        return self.labels, self.label2id, self.id2label
