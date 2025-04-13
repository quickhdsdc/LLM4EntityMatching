from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
import numpy as np
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          DataCollatorForTokenClassification,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          AutoModelForSequenceClassification,
                          pipeline,
                          logging,
                          set_seed)

import evaluate
accuracy = evaluate.load("accuracy")


def collate_fn(examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = torch.stack([torch.as_tensor(example["input_ids"]) for example in examples])
    attention_masks = torch.stack([torch.as_tensor(example["attention_mask"]) for example in examples])
    input_ids = torch.squeeze(input_ids, dim=1).to(device)
    attention_masks = torch.squeeze(attention_masks, dim=1).to(device)
    labels = torch.stack([torch.as_tensor(example["label"]) for example in examples]).to(device)
    # print("########## device",input_ids.device)
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels}


def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class ModelFinetuner:
    def __init__(self) -> None:
        ''

    
    def fine_tune(self, 
                  model,
                  tokenizer,
                  train_ds,
                  val_ds,
                  per_device_train_batch_size,
                  output_dir,
                  train_epochs,
                  learning_rate = 2e-4):
        print('fine-tuning....')
       
        
        args = TrainingArguments(
                output_dir = output_dir,
                num_train_epochs=train_epochs,
                per_device_train_batch_size = per_device_train_batch_size,
                # per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps = 8,
                learning_rate = learning_rate, #learning rate, based on QLoRA paper
                logging_steps=10,
                fp16 = False,
                weight_decay=0.001, # 0.01
                # warmup_steps=500,
                adam_epsilon = 1e-6,
                max_grad_norm=0.3, # 1                       # max gradient norm based on QLoRA paper
                max_steps=-1,
                warmup_ratio=0.03, #0.06                       # warmup ratio based on QLoRA paper
                group_by_length=True,
                lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                report_to="tensorboard",                  # report metrics to tensorboard
                # evaluation_strategy="epoch",              # save checkpoint every epoch
                save_strategy="epoch",
                gradient_checkpointing=True,              # use gradient checkpointing to save memory
                optim="adamw_torch_fused",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                # load_best_model_at_end=True,
                # metric_for_best_model="eval_accuracy",
                 )

        print('train_ds[0]',train_ds[0])
        print('model.config.id2label', model.config.id2label)
        print('model.config.label2id', model.config.label2id)
        trainer = Trainer(
                model=model,
                args=args,
                data_collator=collate_fn,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                # eval_dataset=val_ds,
                # compute_metrics=compute_metrics,
                
            )
       

        model.config.use_cache = False
        do_train = True

        # Launch training and log metrics
        print("Training...")

        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            # trainer.log_metrics("train", metrics)
            # trainer.save_metrics("train", metrics)
            #trainer.save_state()
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            # print(metrics)
            
        # Free memory for merging weights
        del model
        del trainer
        torch.cuda.empty_cache()


        

def main():
    model_finetuner = ModelFinetuner()

if __name__ == "__main__":
    main()