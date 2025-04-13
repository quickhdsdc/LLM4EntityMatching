import torch
from pathlib import Path
from aas_loader import process_all_xml_files
import shutil
import collections
import gc


def run_experiments(args):
    print('task_name: ', args.task_name)
    print('method: ', args.method)
    print('####### 1.load task dataset....')
    from tasks.task_data_loader import TaskDataLoader
    task_data_loader = TaskDataLoader(task_name=args.task_name, train_type=args.train_type, val_type=args.val_type, test_type=args.test_type, imbalance_ratio=args.imbalance_ratio)
    train_ds, val_ds, test_ds = task_data_loader.load_data()
    labels, label2id, id2label = task_data_loader.get_labels() 
    from tasks.task_data_loader import TaskDataLoader_pairwise
    task_data_loader = TaskDataLoader_pairwise(task_name=args.task_name, train_type=args.train_type + '_pairwise', val_type=args.val_type + '_pairwise', test_type=args.test_type + '_pairwise', imbalance_ratio=args.imbalance_ratio)
    _, _, test_ds_pairwise = task_data_loader.load_data()

    print('####### 2.load model and tokenizer....')
    model_name_version_dic = {
                            "mistral-base":{"model_name":"mistralai-Mistral", "model_version":"7B-Instruct-v0.2", "model_root_path": "mistralai/Mistral-7B-Instruct-v0.3"},
                            "mistral-embedding":{"model_name":"Mistral-Embedding", "model_version":"7B-Embedding", "model_root_path": "Salesforce/SFR-Embedding-Mistral"},
                            "llama2-7b":{"model_name":"llama-2-hf", "model_version":"7B", "model_root_path": "meta-llama/Llama-2-7b-hf"},
                            "llama2-7b-chat": {"model_name":"llama-2-hf", "model_version":"7B-Chat", "model_root_path": "meta-llama/Llama-2-7b-chat-hf"},
                            "llama2-13b": {"model_name":"llama-2-hf", "model_version":"13B", "model_root_path": "meta-llama/Llama-2-13b-hf"},
                            "llama2-13b-chat": {"model_name":"llama-2-hf", "model_version":"13B-Chat", "model_root_path": "meta-llama/Llama-2-13b-chat-hf"},
                            "llama3-8b-Instruct": {"model_name":"llama3-8b-Instruct", "model_version":"8b-Instruct", "model_root_path": "meta-llama/Meta-Llama-3-8B-Instruct"},
                            "llama3-8b": {"model_name":"llama3-8b", "model_version":"8b", "model_root_path": "meta-llama/Meta-Llama-3-8B"},
                            "gpt4": {"model_name": "gpt-4", "model_version": "4-1106-preview", "model_root_path": "gpt-4o-2024-05-13"},
                            "repllama-7b": {"model_name": "repllama-7b", "model_version": "v1","model_root_path": "castorini/repllama-v1-7b-lora-doc"},
                            "roberta-large": {"model_name": "roberta-large", "model_version": "large", "model_root_path": "FacebookAI/roberta-large"},
                            "roberta-base": {"model_name": "roberta-base", "model_version": "base", "model_root_path": "FacebookAI/roberta-base"},
                            "all-mpnet-base-v2": {"model_name":"all-mpnet-base-v2", "model_version":"v2", "model_root_path": "sentence-transformers/all-mpnet-base-v2"},
                            "gpt-embedding": {"model_name": "text-embedding-ada", "model_version": "v2", "model_root_path": "text-embedding-ada-002-2"},
                            "BM25": {"model_name": "BM25", "model_version": "v0", "model_root_path": "BM25"},
                            "Linq-Embed":{"model_name":"Linq-Embed-Mistral", "model_version":"7B-Embedding", "model_root_path": "Linq-AI-Research/Linq-Embed-Mistral"},
                            }


    model_name = model_name_version_dic[args.model_dic_key]["model_name"]
    model_version = model_name_version_dic[args.model_dic_key]["model_version"]
    model_root_path = model_name_version_dic[args.model_dic_key]["model_root_path"]

    from tasks.task_model_loader import TaskModelLoader
    model_loader = TaskModelLoader(task_name=args.task_name, method=args.method).model_loader
    if args.method in ['finetuning_llm_c']:
        model, tokenizer = model_loader.load_model_from_path_name_version(model_root_path, model_name, model_version, labels, label2id, id2label, pooling_type='orig', device_map=args.device_map)
    elif args.method in ['finetuning_llm']:
        model, tokenizer = model_loader.load_model_from_path_name_version(model_root_path, model_name, model_version, device_map=args.device_map)
    elif args.method in ['evaluate_gpt']:
        model = model_loader.load_model_from_path_name_version()
        tokenizer = None
    elif args.method in ['finetuning_roberta_c']:
        model, tokenizer = model_loader.load_model_from_path_name_version(model_root_path, labels, label2id, id2label, device_map=args.device_map)


    print('####### 3.preprocess dataset with model and tokenizer....')
    from tasks.task_data_preprocessor import TaskDataPreprocessor
    data_preprocessor_pairwise = TaskDataPreprocessor(task_name=args.task_name, method=args.method).data_preprocessor_pairwise

    if 'inst' in args.input_type:
        prompt_type = "pt0"
    elif 'ICL' in args.input_type:
        prompt_type = "pt1"
    elif 'CoT' in args.input_type:
        prompt_type = "pt2"        
    else:
         prompt_type = None

    if 'st' in args.input_type:
        prompt_st_type = 'st'
    else: 
        prompt_st_type = 'nl'

    if args.method in ['finetuning_llm_c','finetuning_roberta_c']:     
        test_ds_pairwise = data_preprocessor_pairwise.preprocess_data(test_ds_pairwise, label2id, tokenizer, args=args, remove_unused_col=False, is_train=False)
        train_ds = data_preprocessor_pairwise.preprocess_data(train_ds, label2id, tokenizer, args=args,  remove_unused_col=False, is_train=True)
        val_ds = data_preprocessor_pairwise.preprocess_data(val_ds, label2id, tokenizer, args=args, remove_unused_col=False, is_train=False)
        test_ds = data_preprocessor_pairwise.preprocess_data(test_ds, label2id, tokenizer, args=args, remove_unused_col=False, is_train=False)
    elif args.method in ['finetuning_llm']:     
        test_ds_pairwise, response_key = data_preprocessor_pairwise.preprocess_data(tokenizer, test_ds_pairwise, args, False, prompt_type, prompt_st_type)
        train_ds, response_key  = data_preprocessor_pairwise.preprocess_data(tokenizer, train_ds, args, True, prompt_type, prompt_st_type)
        val_ds, response_key  = data_preprocessor_pairwise.preprocess_data(tokenizer, val_ds, args, False, prompt_type, prompt_st_type)
        test_ds, response_key  = data_preprocessor_pairwise.preprocess_data(tokenizer, test_ds, args, False, prompt_type, prompt_st_type)
    elif args.method in ['evaluate_gpt']:
        test_ds = data_preprocessor_pairwise.preprocess_data(model, test_ds, args, False, prompt_type, prompt_st_type)
        test_ds_pairwise = data_preprocessor_pairwise.preprocess_data(model, test_ds_pairwise, args, False, prompt_type, prompt_st_type)
            
    # Task type
    if args.method == 'finetuning_llm':
        task_type = "CAUSAL_LM"
    elif args.method == 'finetuning_llm_c':
        task_type = "SEQ_CLS"
    elif "retrieval" in args.method or 'selection' in args.method:
        task_type = "FEATURE_EXTRACTION"
    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    # create model dir
    # create model dir
    output_dir = Path("./results")
    if not output_dir.exists():
        output_dir.mkdir()
    output_dir = output_dir/args.task_name
    if not output_dir.exists():
        output_dir.mkdir()
    output_dir = output_dir/args.method
    if not output_dir.exists():
        output_dir.mkdir()
    if args.method in ['evaluate_gpt', 'selection_gpt']:
        model_folder_name = model_name
    elif args.method in ['finetuning_llm_c','finetuning_llm','retrieval_llm_c','selection_llm','finetuning_roberta_c']:
        if args.flag_fine_tuning:
            model_folder_name = model_name.split('-')[0] + '-' + model_version + f"_imb{args.imbalance_ratio}_lora-r{args.lora_r}-a{args.lora_alpha}_batch{args.per_device_train_batch_size}_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"
        else:    
            model_folder_name = model_name.split('-')[0] + '-' + model_version + f"_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"+'_orig'
    output_dir = output_dir/model_folder_name
    if not output_dir.exists():
        output_dir.mkdir()
    print('output_dir: ', output_dir)

    if args.flag_fine_tuning and 'gpt' not in args.method:
        print('####### 4.fine-tune model....')
        from tasks.task_model_finetuner import TaskModelFinetuner
        model_finetuner = TaskModelFinetuner(task_name=args.task_name, method=args.method).model_finetuner
        if args.method in ['finetuning_roberta_c']:
            model_finetuner.fine_tune(model, tokenizer, train_ds, val_ds, args.per_device_train_batch_size, output_dir, args.train_epochs, learning_rate=args.learning_rate)
        else:
            model_finetuner.fine_tune(model, tokenizer, train_ds, val_ds,
                                  args.lora_r, args.lora_alpha, args.lora_dropout, args.bias, task_type,
                                  args.per_device_train_batch_size, output_dir, args.train_epochs,
                                  target_modules=args.target_modules, learning_rate=args.learning_rate)

    # inference
    from tasks.task_evaluater import TaskEvaluater
    Evaluater = TaskEvaluater(task_name=args.task_name, method=args.method).evaluater

    file_inference = 'inference.jsonl'
    if not args.flag_fine_tuning or 'gpt' in args.method:
        if args.eval_type == 'pairwise_old':
            if args.method in ['finetuning_llm']:
                Evaluater.evaluate(test=test_ds, labels=labels, label2id=label2id, id2label=id2label, output_dir=output_dir, response_key=response_key, eval_type=args.eval_type, model=model, tokenizer=tokenizer, flag_fine_tuning=args.flag_fine_tuning)
            else:
                Evaluater.evaluate(test=test_ds, labels=labels, label2id=label2id, id2label=id2label, model_dir=output_dir, model=model, tokenizer=tokenizer, max_length=args.max_length, eval_type=args.eval_type, flag_fine_tuning=args.flag_fine_tuning)
        else:
            if args.method in ['finetuning_llm']:
                Evaluater.evaluate(test=test_ds_pairwise, labels=labels, label2id=label2id, id2label=id2label, output_dir=output_dir, response_key=response_key, eval_type=args.eval_type, model=model, tokenizer=tokenizer, flag_fine_tuning=args.flag_fine_tuning)
            else:            
                Evaluater.evaluate(test=test_ds_pairwise, labels=labels, label2id=label2id, id2label=id2label, model_dir=output_dir, model=model, tokenizer=tokenizer, max_length=args.max_length, eval_type=args.eval_type, flag_fine_tuning=args.flag_fine_tuning)
             
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint"):
            print(f"Evaluating {d}")
            if output_dir is not None:
                dir = output_dir / d.name
                if args.eval_type == 'pairwise_old':
                    if args.method in ['finetuning_llm']:
                        Evaluater.evaluate(test=test_ds, labels=labels, label2id=label2id, id2label=id2label, output_dir=dir, response_key=response_key, eval_type=args.eval_type, model=model, tokenizer=tokenizer, flag_fine_tuning=args.flag_fine_tuning)
                    else: 
                        Evaluater.evaluate(test=test_ds, labels=labels, label2id=label2id, id2label=id2label, model_dir=dir, model=model, tokenizer=tokenizer, max_length=args.max_length, eval_type=args.eval_type, flag_fine_tuning=args.flag_fine_tuning)
                else:
                    if args.method in ['finetuning_llm']:
                        Evaluater.evaluate(test=test_ds_pairwise, labels=labels, label2id=label2id, id2label=id2label, output_dir=dir, response_key=response_key, eval_type=args.eval_type, model=model, tokenizer=tokenizer, flag_fine_tuning=args.flag_fine_tuning)
                    else: 
                        Evaluater.evaluate(test=test_ds_pairwise, labels=labels, label2id=label2id, id2label=id2label, model_dir=dir, model=model, tokenizer=tokenizer, max_length=args.max_length, eval_type=args.eval_type, flag_fine_tuning=args.flag_fine_tuning)

if __name__ == "__main__":

    args_dict = {
        # dataset parameters
        'task_name': 'Textual_Abt-Buy',  # Options: 'Structured_Amazon-Google', 'Structured_Walmart-Amazon', 'Textual_Abt-Buy', 'AAS_ECLASS_new'
        'task_name_test': None,
        'train_type': 'train_df',
        'val_type': 'valid_df',
        'test_type': 'test_df',
        'imbalance_ratio': 3,
        # method parameters
        'method': 'evaluate_gpt', # evaluate_gpt, finetuning_llm, finetuning_llm_c, finetuning_roberta_c
        'model_dic_key': 'gpt4', # roberta-large, llama2-13b, llama2-13b-chat, gpt4
        'input_type': 'inst_text_st_on', # 'text_st_on', 'text_on', 'inst_text_st_on', 'ICL_text_st_on', 'CoT_text_st_on'
        'device_map': "auto",
        'max_length': 512, 
        'num_neg': 9,
        # LoRA parameters
        'lora_r': 128,
        'lora_alpha': 128,
        'target_modules': "all-linear",
        # Training parameters
        'learning_rate': 2e-4, # 3e-5 fpr PLM, 2e-4 for LLM
        'lora_dropout': 0.1,
        'bias': "none",
        'per_device_train_batch_size': 32,
        'train_epochs': 10,
        'flag_fine_tuning': True,
        'create_dir': False,
        'eval_type': "retrieval",  # "retrieval", "pairwise_old", "pairwise_new"
    }
    args = collections.namedtuple("args", args_dict.keys())(*args_dict.values())
    run_experiments(args)
    gc.collect()
    torch.cuda.empty_cache()