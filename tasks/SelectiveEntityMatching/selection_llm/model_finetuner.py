import torch
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import numpy as np
from trl import SFTTrainer
from transformers import TrainingArguments
import evaluate
accuracy = evaluate.load("accuracy")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def collate_fn(examples):
    # max_len_query = 0
    # max_len_instr = 0
    # max_len_candidates = 0

    # Calculate the maximum effective lengths using attention masks
    # for example in examples:
    #     max_len_query = max(max_len_query, sum(example['attention_mask_query'][0]))
    #     max_len_instr = max(max_len_instr, sum(example['attention_mask_instr'][0]))
    #     for neg_mask in example['attention_mask_candidates']:
    #         max_len_candidates = max(max_len_candidates, sum(neg_mask))

    # Print the maximum effective lengths
    # print(f"max_len_query: {max_len_query}")
    # print(f"max_len_instr: {max_len_instr}")
    # print(f"max_len_candidates: {max_len_candidates}")
    # print(f"num of candidate samples: {len(example['attention_mask_candidates'])}")
        
    for example in examples:
        example['input_ids'] = torch.as_tensor(example['input_ids_query'])
        example['attention_mask'] = torch.as_tensor(example['attention_mask_query'])

        example['input_ids_instr'] = torch.as_tensor(example['input_ids_instr'])
        example['attention_mask_instr'] = torch.as_tensor(example['attention_mask_instr'])

        example['input_ids_candidates'] = [torch.as_tensor(neg) for neg in example['input_ids_candidates']]
        example['attention_mask_candidates'] = [torch.as_tensor(neg) for neg in example['attention_mask_candidates']]

        example['input_ids_instr'] = torch.as_tensor(example['input_ids_instr'])
        example['attention_mask_instr'] = torch.as_tensor(example['attention_mask_instr'])

        example['labels'] = torch.as_tensor(example['labels'])

    input_ids_query = torch.stack([example["input_ids"].squeeze() for example in examples])
    attention_mask_query = torch.stack([example["attention_mask"].squeeze() for example in examples])

    input_ids_instr = torch.stack([example["input_ids_instr"].squeeze() for example in examples])
    attention_mask_instr = torch.stack([example["attention_mask_instr"].squeeze() for example in examples])

    # Stack the negative samples appropriately
    input_ids_candidates = [torch.stack(example["input_ids_candidates"]).squeeze() for example in examples]
    input_ids_candidates = torch.stack(input_ids_candidates)
    attention_mask_candidates = [torch.stack(example["attention_mask_candidates"]).squeeze() for example in examples]
    attention_mask_candidates = torch.stack(attention_mask_candidates)

    labels = torch.stack([example["labels"].squeeze() for example in examples])

    return {
        "input_ids": input_ids_query,
        "attention_mask": attention_mask_query,
        "input_ids_instr": input_ids_instr,
        "attention_mask_instr": attention_mask_instr,
        "input_ids_candidates": input_ids_candidates,
        "attention_mask_candidates": attention_mask_candidates,
        "labels": labels
    }



class ModelFinetuner:
    def __init__(self) -> None:
        ''

    def print_trainable_parameters(self, model, use_4bit = False):
        """Prints the number of trainable parameters in the model.
        :param model: PEFT model
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        if use_4bit:
            trainable_params /= 2
        print(
            f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
        )

    def find_all_linear_names(self, model):
        """
        Find modules to apply LoRA to.
        :param model: PEFT model
        """
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        print('model.named_modules()', model.named_modules())
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        print(f"LoRA module names: {list(lora_module_names)}")
        return list(lora_module_names)
    
    def create_peft_config(self, r, lora_alpha, target_modules, lora_dropout, bias, task_type):
        """
        Creates Parameter-Efficient Fine-Tuning configuration for the model

        :param r: LoRA attention dimension
        :param lora_alpha: Alpha parameter for LoRA scaling
        :param target_modules: Names of the modules to apply LoRA to
        :param lora_dropout: Dropout Probability for LoRA layers
        :param bias: Specifies if the bias parameters should be trained
        :param task_type: Type of the task
        """
        config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type
        )

        return config
    
    def fine_tune(self, 
                  model,
                  tokenizer,
                  train_ds,
                  val_ds,
                  lora_r,
                  lora_alpha,
                  lora_dropout,
                  bias,
                  task_type,
                  per_device_train_batch_size,
                  #gradient_accumulation_steps,
                  #warmup_steps,
                  #max_steps,
                  #learning_rate,
                  #fp16,
                  #logging_steps,
                  output_dir,
                  #optim, 
                  train_epochs,
                  target_modules="all-linear",
                  learning_rate = 2e-4):
        
        # Prepare the model for training
        model = prepare_model_for_kbit_training(model)
        # Create PEFT configuration for these modules and wrap the model to PEFT
        peft_config = LoraConfig(
            r = lora_r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
        )
        model = get_peft_model(model, peft_config)
        # Print information about the percentage of trainable parameters
        self.print_trainable_parameters(model)
        # Enable gradient checkpointing to reduce memory usage during fine-tuning
        model.gradient_checkpointing_enable()

        args = TrainingArguments(
                output_dir = output_dir,
                num_train_epochs=train_epochs,
                per_device_train_batch_size = per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps = 8,
                learning_rate = learning_rate, #learning rate, based on QLoRA paper
                logging_steps=10,
                fp16 = True,
                weight_decay=0.001,
                max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
                max_steps=-1,
                warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
                group_by_length=True,
                lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                report_to="tensorboard",                  # report metrics to tensorboard
                save_strategy="epoch",
                gradient_checkpointing=True,              # use gradient checkpointing to save memory
                optim="paged_adamw_32bit",
                remove_unused_columns=False,
                evaluation_strategy="no"
                # evaluation_strategy="epoch",              # save checkpoint every epoch
                # load_best_model_at_end=True,
                # metric_for_best_model="eval_recall",
                 )
        
        data_collator = collate_fn
        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                # eval_dataset= val_ds, 
                # compute_metrics=self.compute_metrics,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=4096,
                data_collator = data_collator,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                }
                
            )

        model.config.use_cache = False
        do_train = True

        # Launch training and log metrics
        print("Training...")

        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            #trainer.save_state()
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            print(metrics)

        del model
        del trainer
        torch.cuda.empty_cache()


def main():
    model_finetuner = ModelFinetuner()

if __name__ == "__main__":
    main()