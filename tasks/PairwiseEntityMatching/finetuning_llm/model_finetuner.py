from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
import bitsandbytes as bnb
from trl import SFTTrainer
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

def collate_fn(examples):

    for example in examples:
        example['input_ids'] = torch.as_tensor(example['input_ids_text'])
        example['attention_mask'] = torch.as_tensor(example['attention_mask_text'])
        example['label'] = torch.as_tensor(example['label'])
       
    #print('!!!!!!!!!!!!!!!!', examples[0].keys())
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_masks = torch.stack([example["attention_mask"] for example in examples])
    input_ids = torch.squeeze(input_ids, dim=1)
    attention_masks = torch.squeeze(attention_masks, dim=1)
    labels = torch.stack([example["label"] for example in examples])
    return {"input_ids": input_ids, 
            "attention_mask": attention_masks,
            "labels": labels}



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
            #print('name',name)
            #print('module',module)
            if isinstance(module, cls):
                #print('!!!!!!!!!!!')
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                #print(names)

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
            task_type = task_type,
        )

        return config
    
    def create_training_args(self, 
                             per_device_train_batch_size, 
                             gradient_accumulation_steps, 
                             warmup_steps, 
                             max_steps, 
                             learning_rate, 
                             output_dir, 
                             optim,
                             logging_steps:int =10,
                             fp16:bool=True, ):
        """
        Creates training arguments for the model

        :param per_device_train_batch_size: Batch size per device
        :param gradient_accumulation_steps: Number of gradient accumulation steps
        :param warmup_steps: Number of warmup steps
        :param max_steps: Maximum number of steps
        :param learning_rate: Learning rate
        :param fp16: Specifies if 16-bit precision should be used
        :param logging_steps: Number of logging steps
        :param output_dir: Output directory
        :param optim: Optimizer
        """

        args = TrainingArguments(
                per_device_train_batch_size = per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps = gradient_accumulation_steps,
                learning_rate = learning_rate,
                logging_steps=logging_steps,
                output_dir = output_dir,
                fp16 = fp16,
                warmup_steps = warmup_steps,
                max_steps = max_steps,
                optim = optim,
            ),

        return args

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
                  #fp16,
                  #logging_steps,
                  output_dir,
                  #optim, 
                  train_epochs,
                  target_modules="all-linear",
                  learning_rate = 2e-4
                  ):
        print('fine-tuning....')
        # Enable gradient checkpointing to reduce memory usage during fine-tuning
        model.gradient_checkpointing_enable()
        # Prepare the model for training 
        self.print_trainable_parameters(model)
        model = prepare_model_for_kbit_training(model)
        #self.print_trainable_parameters(model)
        # Get LoRA module names
        print('###2')
        #target_modules = self.find_all_linear_names(model)
        #print(target_modules)
        target_modules = "all-linear"
        # Create PEFT configuration for these modules and wrap the model to PEFT
        print('###3')
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
        #args = self.create_training_args(per_device_train_batch_size, gradient_accumulation_steps, warmup_steps, max_steps, 
        #                                 learning_rate, output_dir, optim, logging_steps, fp16)
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
                evaluation_strategy="epoch",              # save checkpoint every epoch
                save_strategy="epoch",
                #warmup_steps = warmup_steps,
                #max_steps = max_steps,
                gradient_checkpointing=True,              # use gradient checkpointing to save memory
                optim="paged_adamw_32bit",
                #load_best_model_at_end=True,
                #metric_for_best_model="accuracy",
                #evaluation_strategy="epoch",
                #save_strategy="epoch",
                #num_train_epochs=1,
                #load_best_model_at_end=True,
                #metric_for_best_model="accuracy",
                 )
        
        
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        #data_collator = collate_fn
        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                max_seq_length=4096,
                packing=False,
                data_collator = data_collator,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                }
                
            )
        # Training parameters
        '''
        trainer = Trainer(
            model = model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args = args,
            data_collator = data_collator,
            )
        '''

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
            

        # Save model
        #print("Saving last checkpoint of the model...")
        #trainer.model.save_pretrained(output_dir)

        #print('######evaluate....')
        print('Evaluation...')
        #trainer.evaluate(val_ds)


        # Free memory for merging weights
        del model
        del trainer
        torch.cuda.empty_cache()


        

def main():
    model_finetuner = ModelFinetuner()

if __name__ == "__main__":
    main()