import torch
from transformers import TrainingArguments, Trainer

def collate_fn(examples):
        
    for example in examples:
        example['input_ids'] = torch.as_tensor(example['input_ids_query'])
        example['attention_mask'] = torch.as_tensor(example['attention_mask_query'])
        example['input_ids_candidates'] = [torch.as_tensor(neg) for neg in example['input_ids_candidates']]
        example['attention_mask_candidates'] = [torch.as_tensor(neg) for neg in example['attention_mask_candidates']]
        example['labels'] = torch.as_tensor(example['labels'])

    input_ids_query = torch.stack([example["input_ids"].squeeze() for example in examples])
    attention_mask_query = torch.stack([example["attention_mask"].squeeze() for example in examples])

    # Stack the negative samples appropriately
    input_ids_candidates = [torch.stack(example["input_ids_candidates"]).squeeze() for example in examples]
    input_ids_candidates = torch.stack(input_ids_candidates)
    attention_mask_candidates = [torch.stack(example["attention_mask_candidates"]).squeeze() for example in examples]
    attention_mask_candidates = torch.stack(attention_mask_candidates)

    labels = torch.stack([example["labels"].squeeze() for example in examples])

    return {
        "input_ids": input_ids_query,
        "attention_mask": attention_mask_query,
        "input_ids_candidates": input_ids_candidates,
        "attention_mask_candidates": attention_mask_candidates,
        "labels": labels
    }



class ModelFinetuner:
    def __init__(self) -> None:
        ''
    
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
                  output_dir,
                  train_epochs,
                  target_modules="all-linear",
                  learning_rate = 2e-4):


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
                gradient_checkpointing=False,              # use gradient checkpointing to save memory
                optim="adamw_torch_fused",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                # load_best_model_at_end=True,
                # metric_for_best_model="eval_accuracy",
                 )
        
        data_collator = collate_fn
        trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                # eval_dataset= val_ds, 
                # compute_metrics=self.compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
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
