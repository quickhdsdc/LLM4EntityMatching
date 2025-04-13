# LLM4EntityMatching
This is the code repository for our work related to the general entity matching (EM) task and the Asset Administration Shell (AAS)-specific EM task.
## Selective Entity Matching
In the paper "Fine-Tuning Large Language Models with Contrastive Margin Ranking Loss for Selective Entity Matching in Product Data Integration" (submitted), we first revisit the standard pairwise EM setting by recompiling existing benchmark datasets to include more hard negative candidates, which are semantically similar to corresponding query entities. We then evaluate state-of-the-art (SOTA) pairwise matchers on these recompiled datasets, revealing the limitations of the conventional pairwise EM approach under more challenging and realistic conditions. Second, we propose a selective EM approach that formulates EM as a listwise selection task, where the query entity is compared directly with the entire candidate set rather than evaluated through independent pairwise classifications. Accordingly, a new evaluation framework is introduced, including recompiled benchmark datasets and a new evaluation metric. Third, we propose a selective EM method Mistral4SelectEM, which fine-tunes an LLM-based embedding model for selective EM by structuring it into a Siamese network and fine-tuning it with a novel contrastive margin ranking loss (CMRL). It aims to enhance the model’s ability to distinguish true positives from semantically similar negatives. 
### Method
![](/resource/Mistral4SelectEM.png)
*Fig. 3. Illustration of the end-to-end selective EM (left) and the fine-tuning strategy for Mistral4SelectEM (right). The entire process involves a single set of LLM weights, which is presented as the green block. The “red” adapter is the fine-tuned LoRA weights, which are merged with the embedding model in the inference stage for selective EM*<br>  
This method is implemented in /tasks/SelectiveEntityMatching/selection_llm. The comparative methods are implemented in selection_plm, selection_gpt, and cross_selection_llm. The selection of the methods is controlled by the main args.method.
### data
The original EM benchmark datasets—Structured_Amazon-Google, Structured_Walmart-Amazon, and Textual_Abt-Buy—are recompiled by retaining labeled positive pairs from existing splits. For each query entity, we apply a blocking step using the Linq-Embed-Mistral embedding model and FAISS for efficient indexing and retrieval. Entities are encoded into vectors, and the top-K (=10) candidates are ranked by semantic similarity. Labels from the original dataset are retained for matched entities; others are labeled as negatives, ensuring each query has exactly 10 candidates. While most queries have one true match, some may have multiple.  
The original data (fixed train-valid-test split) is still available as train_df.csv, valid_df.csv, test_df.csv. The recompiled dataset is structured in two formats. The first format maintains pairwise data, consistent with the original dataset, for use in pairwise EM methods, named as e.g. test_df_pairwise.csv. The second format adopts an information-retrieval-style representation, where the label for a query is a binary match vector, indicating the position of true positive candidates in the top-K retrieved set. They are named as e.g. test_df_new.csv.
### usage & arguments
EM_data_convert.py is used to convert the original pairwise EM datasets into the recompiled datasets.  
aas_loader.py contains the relevant functions to parse AAS models to extract entities. For the general EM task, this function is not needed.  
data_representation.py contains the relevant functions construct the original EM datasets (from DeepMatcher) in form of dataframe.   
<br>EM_training_contrastive.py is used to conduct the experiments of selective EM. It has the following arguments.  
'task_name': select the EM dataset for experiments,  # Options: 'Structured_Amazon-Google', 'Structured_Walmart-Amazon', 'Textual_Abt-Buy'  
'task_name_test': set for cross-dataset test. For instance, setting task_name_test as 'Structured_Walmart-Amazon', while task_name is 'Structured_Amazon-Google', means training the model on Structured_Amazon-Google, and testing it on Structured_Walmart-Amazon  
'train_type', 'val_type', 'test_type': 'selecting the data to be used # defaults as train_df_new', valid_df_new, test_df_new  
'method': method to be used,  # Options: 'selection_llm', 'cross_selection_llm', 'selection_plm', 'selection_gpt'  
'model_dic_key': 'base model to be fine-tuned, # mistral-embedding, all-mpnet-base-v2, Linq-Embed, gpt4  
'input_type': prompting format,  # Options: 'inst_text_on', 'text_st_on', etc.  
'device_map': "auto",  
'max_length': , # 128 for selection_llm, 1024 for cross_selection_llm (1280 for Textual_Abt-Buy)  
'num_neg': 9,  
'loss_type': 'CMRL',  # Options: 'CMRL', 'InfoNCE', 'Focal'  
'margin': 0.5, # CMRL hyperparameter  
'alpha': 0.6,  
'top_k': 3,  
'lora_r': 64, # Lora parameters  
'lora_alpha': 64,  
'target_modules': "all-linear",  
'learning_rate': 2e-4, # 3e-5 fpr PLM, 2e-4 for LLM  
'lora_dropout': 0.1,  
'bias': "none",    
'per_device_train_batch_size': 32,  
'train_epochs': 10,  
## Pairwise Entity Matching
In the paper "[Dual data mapping with fine-tuned large language models and asset administration shells toward interoperable knowledge representation](https://www.sciencedirect.com/science/article/pii/S0736584524001248?via%3Dihub)", we propose two approaches for fine-tuning pairwise EM.
'flag_fine_tuning': True,  
'create_dir': False,  
'eval_type': "retrieval",  # Options: "retrieval", "pairwise" ("pairwise" means the model is trained by contrastive learning, but inference for pairwise classification. But this does not work well. has been removed.)

## Pairwise Entity Matching
In the paper "Dual data mapping with fine-tuned large language models and asset administration shells toward interoperable knowledge representation" https://www.sciencedirect.com/science/article/pii/S0736584524001248?via%3Dihub
