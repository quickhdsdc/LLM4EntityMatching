from pathlib import Path
import pandas as pd
from data_representation import DeepMatcherProcessor
import os
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import faiss
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from torch import Tensor
import json
from aas_loader import process_all_xml_files


class EmbeddingModelWrapper(ABC):
    @abstractmethod
    def embed_documents(self, texts):
        pass


class AzureEmbeddingModel(EmbeddingModelWrapper):
    def __init__(self):
        self.model = AzureOpenAIEmbeddings(
            model="text-embedding-3-large-1",
            openai_api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_type="azure",
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)


class SentenceTransformerEmbeddingModel(EmbeddingModelWrapper):
    def __init__(self, model_root_path='sentence-transformers/all-mpnet-base-v2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_root_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.model = AutoModel.from_pretrained(model_root_path, device_map=self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts):
        embeddings = []
        self.model.eval()
        batch_size = 256
        if len(texts) <= batch_size:
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            self.model.eval()
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.detach().cpu().numpy())
            sentence_embeddings = np.concatenate(embeddings, axis=0)
        return sentence_embeddings


class MistralEmbeddingModel(EmbeddingModelWrapper):
    def __init__(self, model_root_path=None):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, 
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
            )
        HF_token = os.getenv("HFTOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_root_path, token=HF_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.model = AutoModel.from_pretrained(model_root_path, quantization_config = self.bnb_config, device_map = 'auto')
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.sliding_window = 4096
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        

    def embed_documents(self, texts):
        embeddings = []
        self.model.eval()
        batch_size = 52
        if len(texts) <= batch_size:
            texts_dict = self.tokenizer(texts, max_length=4096, padding="max_length", truncation=True,return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**texts_dict)    
                texts_embeddings = self.last_token_pool(outputs.last_hidden_state, texts_dict['attention_mask'])
                texts_embeddings = F.normalize(texts_embeddings, p=2, dim=1)
        else:
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                batch_dict = self.tokenizer(batch, max_length=4096, padding="max_length", truncation=True,return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**batch_dict)    
                    batch_embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)           
                embeddings.append(batch_embeddings.detach().cpu().numpy())
            texts_embeddings = np.concatenate(embeddings, axis=0)
        return texts_embeddings


def get_embedding_model(model_type='azure', model_root_path=None):
    if model_type == 'azure':
        return AzureEmbeddingModel()
    elif model_type == 'sentenceTransformer':
        return SentenceTransformerEmbeddingModel()
    elif 'Mistral' in model_type or 'Qwen2' in model_type:
        return MistralEmbeddingModel(model_root_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class EntityMatcherProcessor:
    def __init__(self, task_name, task_data_dir, model, model_type):
        self.task_name = task_name
        self.task_data_dir = task_data_dir
        self.processor = DeepMatcherProcessor()
        self.model = model
        self.model_type = model_type
        self.target_corpus = self.load_target_corpus()
        self.target_corpus_embeddings = self.get_or_compute_corpus_embeddings()
        self.faiss_index = self.build_faiss_index()

    def load_data(self, file_type):
        file_path = self.task_data_dir / f"{file_type}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
        else:
            if file_type == 'train_df':
                df = self.processor.get_train_examples(self.task_data_dir)
            elif file_type == 'valid_df':
                df = self.processor.get_valid_examples(self.task_data_dir)
            elif file_type == 'test_df':
                df = self.processor.get_test_examples(self.task_data_dir)
            df.to_csv(file_path, index=False)
        df[['text_src', 'text_tgt']] = df.apply(lambda sample: pd.Series(self.merge_columns(sample)), axis=1)
        return df

    def merge_columns(self, sample):
        # Dynamically find columns ending with '_left' or '_right'
        left_columns = [col for col in sample.index if col.endswith('_left') and col not in ['id_left', 'semanticId_left']]
        right_columns = [col for col in sample.index if col.endswith('_right') and col not in ['id_right', 'shortname_right', 'semanticId_right', 'remark_right']]

        # Concatenate the values for left and right columns
        text_src = '. '.join([str(sample[col]) for col in left_columns])
        text_tgt = '. '.join([str(sample[col]) for col in right_columns])
        # text_src = '<s> '.join([str(sample[col]) for col in left_columns])
        # text_tgt = '<s> '.join([str(sample[col]) for col in right_columns])

        return text_src, text_tgt

    def load_target_corpus(self):
        # Check if path points to a file or a directory
        target_corpus_path = self.task_data_dir / "tableB.csv"
        if target_corpus_path.exists():
            # Case 1: tableB.csv exists
            target_corpus = pd.read_csv(target_corpus_path)
            target_corpus[['text']] = target_corpus.apply(
                lambda sample: pd.Series(self.merge_columns_corpus(sample)), axis=1
            )
            return target_corpus
        else:
            # Case 2: XML-based ECLASS dictionary
            target_corpus_path = self.task_data_dir / "ECLASS12_0_Dictionary_ADVANCED_XML_EN"
            df_eclass = process_all_xml_files(target_corpus_path)
            df_eclass.drop_duplicates(subset='definition', inplace=True, ignore_index=True)
            # Compose the text field
            df_eclass["text"] = df_eclass.apply(
                lambda row: f"{row['name']}. {row['definition']}",
                axis=1
            )
            return df_eclass[["text"]]

    def merge_columns_corpus(self, sample):
        # Dynamically find columns ending with '_left' or '_right' excluding id columns
        columns = [col for col in sample.index if not col.startswith('id')]
        text = '. '.join([str(sample[col]) for col in columns])
        return text

    def get_or_compute_corpus_embeddings(self):
        target_corpus_embeddings_file = self.task_data_dir / f'target_corpus_embeddings_{self.model_type}.pkl'
        if target_corpus_embeddings_file.exists():
            with open(target_corpus_embeddings_file, 'rb') as f:
                target_corpus_embeddings = pickle.load(f)
            print("Loaded precomputed corpus embeddings.")
        else:
            target_corpus_list = self.target_corpus['text'].tolist()
            target_corpus_embeddings = self.model.embed_documents(target_corpus_list)
            with open(target_corpus_embeddings_file, 'wb') as f:
                pickle.dump(target_corpus_embeddings, f)
            print("Saved new corpus embeddings.")
        return np.array(target_corpus_embeddings)

    def build_faiss_index(self):
        faiss_index = faiss.IndexFlatIP(self.target_corpus_embeddings.shape[1])
        faiss_index.add(self.target_corpus_embeddings)
        return faiss_index
    
    def compute_mrr(self, labels_list):
        reciprocal_ranks = []
        for labels in labels_list:
            try:
                rank = labels.index(1) + 1
                reciprocal_ranks.append(1 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def compute_hitrate(self, labels_list, k):
        hits = [1 if 1 in labels[:k] else 0 for labels in labels_list]
        return sum(hits) / len(hits)

    def prepare_candidates(self, df, file_type):
        if 'id_left' not in df.columns:
            df_new = df[df['label'] == 'yes'][[
                'text_src',
                'text_tgt',
                'label'
            ]].copy().reset_index()
        else:
            df_sorted_pos = df[df['label'] == 'yes'].sort_values(by='id_left').reset_index(drop=True)
            # Dynamically find columns ending with '_left' or '_right'
            left_columns = [col for col in df.columns if col.endswith('_left') and col not in ['id_left']]
            right_columns = [col for col in df.columns if col.endswith('_right') and col not in ['id_right']]

            # Group by 'id_left' and keep the required columns
            agg_dict = {col: 'first' for col in left_columns}
            agg_dict.update({col: lambda x: list(x) for col in right_columns})
            agg_dict.update({
                'text_src': 'first',  # Keep the first occurrence of the text_src
                'id_right': lambda x: list(x),  # Aggregate all true targets into a list
                'text_tgt': lambda x: list(x)
            })

            df_new = df_sorted_pos.groupby('id_left').agg(agg_dict).reset_index()

        queries = df_new['text_src'].tolist()
        if 'Mistral' in self.model_type or 'Qwen2' in self.model_type:
            task = 'Instruct: Given a query entity, retrieve similar entity that matches the query'
            queries = [f"{task}\nQuery: {query}" for query in df_new['text_src'].tolist()]

        # Load or compute query embeddings based on the file type
        queries_embeddings_file = self.task_data_dir / f"{file_type}_queries_embeddings_{self.model_type}.pkl"
        if queries_embeddings_file.exists():
            with open(queries_embeddings_file, 'rb') as f:
                queries_embeddings = pickle.load(f)
            print(f"Loaded precomputed query embeddings for {file_type}.")
        else:
            queries_embeddings = self.model.embed_documents(queries)
            with open(queries_embeddings_file, 'wb') as f:
                pickle.dump(queries_embeddings, f)
            print(f"Saved new query embeddings for {file_type}.")
        if isinstance(queries_embeddings, torch.Tensor):
            queries_embeddings = queries_embeddings.detach().cpu().numpy()
        elif isinstance(queries_embeddings, list) and isinstance(queries_embeddings[0], torch.Tensor):
            queries_embeddings = np.array([emb.detach().cpu().numpy() for emb in queries_embeddings])
        else:
            queries_embeddings = np.array(queries_embeddings)

        candidates_list = []
        labels_list = []
        labels_k20_list = []

        for i in tqdm(range(len(queries_embeddings))):
            query_embedding = queries_embeddings[i].reshape(1, -1)
            scores, indices = self.faiss_index.search(query_embedding, 20)
            candidates = [self.target_corpus['text'].iloc[idx] for idx in
                          indices[0]]  # Retrieve corresponding candidates
            truth_list = df_new.at[i, 'text_tgt']
            if isinstance(truth_list, str):
                truth_list = [truth_list]
            label_k20 = [1 if any(truth in candidate for truth in truth_list) else 0 for candidate in candidates]
            label_k10 = label_k20[:10]
            candidates_k10 = candidates[:10]

            # If there is no positive match, replace the last negative sample with the truth
            if 1 not in label_k10:
                candidates_k10[-1] = truth_list[0]  # Replace a random candidate with the first truth
                label_k10[-1] = 1

            candidates_list.append(candidates_k10)
            labels_k20_list.append(label_k20)
            labels_list.append(label_k10)

        mrr = self.compute_mrr(labels_list)
        hitrate_5 = self.compute_hitrate(labels_k20_list, 5)
        hitrate_10 = self.compute_hitrate(labels_k20_list, 10)
        hitrate_20 = self.compute_hitrate(labels_k20_list, 20)
        report = {
            'num_query': len(labels_list),
            'Mean Reciprocal Rank top10': mrr,
            'HitRate@5': hitrate_5,
            'HitRate@10': hitrate_10,
            'HitRate@20': hitrate_20,
        }
        with open(self.task_data_dir / f"inference_report_{self.model_type}_{file_type}.json", 'w') as f:
            f.write(json.dumps(report))

        df_new['candidates'] = candidates_list
        df_new['labels'] = labels_list
        return df_new

    def process_and_save(self, file_type):
        df = self.load_data(file_type)
        df_new = self.prepare_candidates(df, file_type)
        df_new.to_csv(self.task_data_dir / f"{file_type}_new.csv", index=False)

    def prepare_candidates_pairwise(self, df, file_type):
        df_sorted_pos = df[df['label'] == 'yes'].sort_values(by='id_left').reset_index(drop=True)

        # Dynamically find columns ending with '_left' or '_right'
        left_columns = [col for col in df.columns if col.endswith('_left') and col not in ['id_left']]
        right_columns = [col for col in df.columns if col.endswith('_right') and col not in ['id_right']]

        # Group by 'id_left' and keep the required columns
        agg_dict = {col: 'first' for col in left_columns}
        agg_dict.update({col: lambda x: list(x) for col in right_columns})
        agg_dict.update({
            'text_src': 'first',  # Keep the first occurrence of the text_src
            'id_right': lambda x: list(x),  # Aggregate all true targets into a list
            'text_tgt': lambda x: list(x)
        })

        df_new = df_sorted_pos.groupby('id_left').agg(agg_dict).reset_index()

        queries = df_new['text_src'].tolist()
        if 'Mistral' in self.model_type or 'Qwen2' in self.model_type:
            task = 'Instruct: Given a query entity, retrieve similar entity that matches the query'
            queries = [f"{task}\nQuery: {query}" for query in df_new['text_src'].tolist()]
        # Load or compute query embeddings based on the file type
        queries_embeddings_file = self.task_data_dir / f"{file_type}_queries_embeddings_{self.model_type}.pkl"
        if queries_embeddings_file.exists():
            with open(queries_embeddings_file, 'rb') as f:
                queries_embeddings = pickle.load(f)
            print(f"Loaded precomputed query embeddings for {file_type}.")
        else:
            queries_embeddings = self.model.embed_documents(queries)
            with open(queries_embeddings_file, 'wb') as f:
                pickle.dump(queries_embeddings, f)
            print(f"Saved new query embeddings for {file_type}.")
        if isinstance(queries_embeddings, torch.Tensor):
            queries_embeddings = queries_embeddings.detach().cpu().numpy()
        elif isinstance(queries_embeddings, list) and isinstance(queries_embeddings[0], torch.Tensor):
            queries_embeddings = np.array([emb.detach().cpu().numpy() for emb in queries_embeddings])
        else:
            queries_embeddings = np.array(queries_embeddings)

        # Prepare lists to hold pairwise data
        pairwise_data = []
        if 'train' in file_type:
            guid_prefix = 'train'
        elif 'valid' in file_type:
            guid_prefix = 'valid'
        elif 'test' in file_type:
            guid_prefix = 'test'
        else:
            guid_prefix = 'unknown'
        for i in tqdm(range(len(queries_embeddings))):
            query_embedding = queries_embeddings[i].reshape(1, -1)
            scores, indices = self.faiss_index.search(query_embedding, 10)

            # candidate_indices = negative_indices + true_indices
            truth_list = df_new.at[i, 'text_tgt']
            true_indices = [idx for idx in indices[0] if self.target_corpus['text'].iloc[idx] in truth_list]
            candidate_indices = indices[0]

            # For each candidate, create a pairwise entry with the query
            for idx in candidate_indices:
                candidate_id = self.target_corpus.at[idx, 'id']

                # Extract the attributes for the candidate entity
                candidate_data = {
                    f"{col}_right": self.target_corpus.at[idx, col] for col in self.target_corpus.columns if
                    col not in ['id','text']
                }

                # Prepare the complete pair data
                pair_data = {
                    'id_left': df_new.at[i, 'id_left'],
                    **{col: df_new.at[i, col] for col in left_columns},
                    'id_right': candidate_id,
                    **candidate_data,
                    'guid': f"{guid_prefix}-{df_new.at[i, 'id_left']}-{candidate_id}",
                    'label': 1 if idx in true_indices else 0
                }
                pairwise_data.append(pair_data)

        # Create a new DataFrame for pairwise data
        df_pairwise = pd.DataFrame(pairwise_data)
        return df_pairwise

    def process_and_save_pairwise(self, file_type):
        df = self.load_data(file_type)
        df_pairwise = self.prepare_candidates_pairwise(df, file_type)
        df_pairwise.to_csv(self.task_data_dir / f"{file_type}_pairwise.csv", index=False)


# Load environment variables
load_dotenv()
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI resource's endpoint value.
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
model_type = 'Mistral-Ling'  # 'azure', 'sentenceTransformer', 'Mistral-Ling', 'Mistral-SFR', 'Qwen2'
model_root_path = 'Linq-AI-Research/Linq-Embed-Mistral' # 'Linq-AI-Research/Linq-Embed-Mistral', 'Salesforce/SFR-Embedding-Mistral', 'Alibaba-NLP/gte-Qwen2-7B-instruct'
model = get_embedding_model(model_type, model_root_path)
# Create an instance of EntityMatcherProcessor
task_name = "Textual_Abt-Buy" # "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Textual_Abt-Buy", "AAS_ECLASS_new"
train_type = 'train_df'
val_type = 'valid_df'
test_type = 'test_df'
task_data_dir = Path('data/_entity matching') / task_name
entity_matcher_processor = EntityMatcherProcessor(task_name, task_data_dir, model, model_type)

# Process and save train, valid, and test DataFrames
entity_matcher_processor.process_and_save('train_df')
entity_matcher_processor.process_and_save('valid_df')
entity_matcher_processor.process_and_save('test_df')
entity_matcher_processor.process_and_save_pairwise('train_df')
entity_matcher_processor.process_and_save_pairwise('valid_df')
entity_matcher_processor.process_and_save_pairwise('test_df')