import os
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset


from typing import Callable, Dict, List, Any

import time
import json
import transformers
import accelerate
import peft
import huggingface_hub
from huggingface_hub import notebook_login
print(f"Huggingface Hub version: {huggingface_hub.__version__}")

from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
import evaluate
from archiv.model_builder import ModelBuilder
#from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          AutoModelForSequenceClassification,
                          pipeline,
                          logging,
                          set_seed)
import os
from random import randrange
from functools import partial
#from trl import SFTTrainer
#from google.colab import drive
#drive.mount('/content/drive')

class TaskDataLoader:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str) -> None:
        '''
        params:task_name: str: name of the task
        params:train_type: str: type of the training data in ['train','train_small']
        params:val_type: str: type of the validation data in ['test']
        params:test_type: str: type of the test data in ['test','test_big']
        '''
        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.task_data_dir = Path('data/Re3-Sci/tasks') / task_name
        data_files = {}
        for i in data_files.iterdir():
            if i.is_file() and i.name.endswith('.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files)
        print('dataset', self.dataset)
        self.labels = self.dataset['train'].features['label'].names
        print('labels', self.labels)
            


def main():
    task_data_loader = TaskDataLoader('edit_intent_classification', 'train', 'test', 'test_big')

if __name__ == "__main__":
    main()
