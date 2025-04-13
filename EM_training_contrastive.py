import torch
from pathlib import Path
import shutil
import collections
import gc


def run_experiments(args):
    print('task_name: ', args.task_name)
    print('method: ', args.method)
    print('####### 1.load task dataset....')
    from tasks.task_data_loader import TaskDataLoader_retrieval
    task_data_loader_retrieval = TaskDataLoader_retrieval(task_name=args.task_name, train_type=args.train_type, val_type=args.val_type, test_type=args.test_type, task_name_test = args.task_name_test)
    train_ds, val_ds, test_ds = task_data_loader_retrieval.load_data()

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
    if args.method in ['selection_plm']:
        model, tokenizer = model_loader.load_model_from_path_name_version(model_root_path, args)
    elif args.method in ['selection_gpt']:
        model = model_loader.load_model_from_path_name_version()
        tokenizer = None
    elif args.method in ['selection_llm', 'cross_selection_llm']:
        model, tokenizer = model_loader.load_model_from_path_name_version(model_root_path, args, args.device_map)

    print('####### 3.preprocess dataset with model and tokenizer....')
    from tasks.task_data_preprocessor import TaskDataPreprocessor
    data_preprocessor = TaskDataPreprocessor(task_name=args.task_name, method=args.method).data_preprocessor

    if args.method in ['selection_gpt']:
        test_ds = data_preprocessor.preprocess_data(test_ds, input_type=args.input_type, max_length=args.max_length,
                                                    num_neg=args.num_neg, remove_unused_col=False, is_train=False)
    elif args.method in ['selection_llm', 'cross_selection_llm', 'selection_plm']:
        train_ds = data_preprocessor.preprocess_data(train_ds, tokenizer, input_type=args.input_type, max_length=args.max_length, num_neg=args.num_neg, remove_unused_col=False, is_train=True)
        val_ds = data_preprocessor.preprocess_data(val_ds, tokenizer, input_type=args.input_type, max_length=args.max_length, num_neg=args.num_neg, remove_unused_col=False, is_train=False)
        test_ds = data_preprocessor.preprocess_data(test_ds, tokenizer, input_type=args.input_type, max_length=args.max_length, num_neg=args.num_neg, remove_unused_col=False, is_train=False)
            
    # Task type
    #task_type = None #"SEQ_CLS" (for default sequence classification, finetuning_llm_c) #"CAUSAL_LM" (for defaut text generation, finetuning_llm) # None (for finetuning_llm_c_siamese)
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
    output_dir = Path("./results")
    if not output_dir.exists():
        output_dir.mkdir()
    output_dir = output_dir/args.task_name
    if not output_dir.exists():
        output_dir.mkdir()
    output_dir = output_dir/args.method
    if not output_dir.exists():
        output_dir.mkdir()
    if args.method in ['selection_gpt']:
        model_folder_name = model_name
    elif args.method in ['selection_plm', 'selection_llm', 'cross_selection_llm']:
        if args.task_name_test == None:
            if args.flag_fine_tuning:
                if args.loss_type == 'CMRL':
                    model_folder_name = model_name.split('-')[0] + '-' + model_version + '-' + args.loss_type + f"_m{args.margin}a{args.alpha}H{args.top_k}" + f"_lora-r{args.lora_r}-a{args.lora_alpha}_batch{args.per_device_train_batch_size}_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"
                else:
                    model_folder_name = model_name.split('-')[0] + '-' + model_version + '-' + args.loss_type + f"_lora-r{args.lora_r}-a{args.lora_alpha}_batch{args.per_device_train_batch_size}_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"
            else:    
                model_folder_name = model_name.split('-')[0] + '-' + model_version + f"_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"+'_orig'
        else:
            if args.flag_fine_tuning:
                if args.loss_type == 'CMRL':
                    model_folder_name = model_name.split('-')[0] + '-' + model_version + '-' + args.task_name_test + '-' + args.loss_type + f"_m{args.margin}a{args.alpha}H{args.top_k}" + f"_lora-r{args.lora_r}-a{args.lora_alpha}_batch{args.per_device_train_batch_size}_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"
                else:
                    model_folder_name = model_name.split('-')[0] + '-' + model_version + '-' + args.task_name_test + '-' + args.loss_type + f"_lora-r{args.lora_r}-a{args.lora_alpha}_batch{args.per_device_train_batch_size}_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"
            else:    
                model_folder_name = model_name.split('-')[0] + '-' + model_version + '-' + args.task_name_test + f"_neg{args.num_neg}_max_length{args.max_length}_{args.input_type}"+'_orig'
        output_dir = output_dir/model_folder_name
    if not output_dir.exists():
        output_dir.mkdir()
    if args.create_dir:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
    print('output_dir: ', output_dir)

    if args.flag_fine_tuning and 'gpt' not in args.method:
        print('####### 4.fine-tune model....')
        from tasks.task_model_finetuner import TaskModelFinetuner
        model_finetuner = TaskModelFinetuner(task_name=args.task_name, method=args.method).model_finetuner
        model_finetuner.fine_tune(model, tokenizer, train_ds, val_ds,
                                  args.lora_r, args.lora_alpha, args.lora_dropout, args.bias, task_type,
                                  args.per_device_train_batch_size, output_dir, args.train_epochs,
                                  target_modules=args.target_modules, learning_rate=args.learning_rate)

    # inference
    from tasks.task_evaluater import TaskEvaluater
    Evaluater = TaskEvaluater(task_name=args.task_name, method=args.method).evaluater

    file_inference = 'inference.jsonl'
    if not args.flag_fine_tuning or 'gpt' in args.method:
        Evaluater.evaluate(test=test_ds, model_dir=output_dir, output_file=file_inference, model=model, tokenizer=tokenizer, max_length=args.max_length, flag_fine_tuning=args.flag_fine_tuning, num_neg=args.num_neg, batch_size=args.per_device_train_batch_size, threshold=0.8, eval_type=args.eval_type)
            
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint"):
            print(f"Evaluating {d}")
            if output_dir is not None:
                dir = output_dir / d.name
                Evaluater.evaluate(test=test_ds, model_dir=dir, output_file=file_inference, model=model, tokenizer=tokenizer, max_length=args.max_length, flag_fine_tuning=args.flag_fine_tuning, num_neg=args.num_neg, batch_size=args.per_device_train_batch_size, threshold=0.8, eval_type=args.eval_type)

if __name__ == "__main__":

    args_dict = {
        # dataset parameters
        'task_name': 'Structured_Amazon-Google',  # Options: 'Structured_Amazon-Google', 'Structured_Walmart-Amazon', 'Textual_Abt-Buy', 'AAS_ECLASS_new'
        'task_name_test': 'Textual_Abt-Buy',
        'train_type': 'train_df_new',
        'val_type': 'valid_df_new',
        'test_type': 'test_df_new',
        # method parameters
        'method': 'selection_llm',  # Options: 'selection_llm', 'cross_selection_llm', 'selection_plm', 'selection_gpt'
        'model_dic_key': 'Linq-Embed', # mistral-embedding, all-mpnet-base-v2, Linq-Embed, gpt4
        'input_type': 'inst_text_on',  # Options: 'inst_text_on', 'text_st_on', etc.
        'device_map': "auto",
        'max_length': 128, #1024 when cross_selection_llm, 1200 for Textual_Abt-Buy
        'num_neg': 9,
        'loss_type': 'CMRL',  # Options: 'CMRL', 'InfoNCE', 'Focal'
        'margin': 0.5,
        'alpha': 0.6,
        'top_k': 1,
        # LoRA parameters
        'lora_r': 64,
        'lora_alpha': 64,
        'target_modules': "all-linear",
        # Training parameters
        'learning_rate': 2e-4, # 3e-5 fpr PLM, 2e-4 for LLM
        'lora_dropout': 0.1,
        'bias': "none",
        'per_device_train_batch_size': 32,
        'train_epochs': 10,
        'flag_fine_tuning': True,
        'create_dir': False,
        'eval_type': "retrieval",  # Options: "retrieval", "pairwise"
    }
    args = collections.namedtuple("args", args_dict.keys())(*args_dict.values())
    run_experiments(args)
    gc.collect()
    torch.cuda.empty_cache()