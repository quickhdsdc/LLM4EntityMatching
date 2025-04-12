from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from peft import PeftModelForFeatureExtraction
from transformers import AutoTokenizer
import torch.nn.functional as F
from .modelling_llama import EntityRetrieverMistral
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import f1_score
import os

class Evaluater:
    def __init__(self) -> None:
        print('Evaluating the model...')

    def merge_model(self, model, finetuned_model_dir:Path, eval_type:str):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        compute_dtype = getattr(torch, "float16")   
        model =  PeftModelForFeatureExtraction.from_pretrained(
                        model=model,
                        model_id=str(finetuned_model_dir)+'/',
                        torch_dtype=compute_dtype,
                        return_dict=False,
                        low_cpu_mem_usage=True,
                        device_map='auto',
                        output_hidden_states=True
                    )
        model = model.merge_and_unload()
        encoder = model.encoder
        score = model.score
        model = model.model
        model = EntityRetrieverMistral(model.config, model=model, score=score, encoder=encoder)
        model.config.sliding_window = 4096
        print('model has been merged')
        model.config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer


    def collate_fn(self, examples):

        for idx, example in enumerate(examples):
            try:
                example['input_ids'] = torch.as_tensor(example['input_ids_text']).squeeze(0)
                example['attention_mask'] = torch.as_tensor(example['attention_mask_text']).squeeze(0)
                example['labels'] = torch.as_tensor(example['labels'])
            except Exception as e:
                print(f"Error processing example {idx}: {e}")

        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])

        labels = torch.stack([example["labels"] for example in examples])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    
    def calculate_mrr_with_penalty(self, ranked_indices, true_labels):
        """
        Calculate MRR with penalty for false positives and missed detections.
        Args:
        - ranked_indices: The list of ranked candidate indices after prediction.
        - true_labels: The ground truth labels (list of 0s and 1s).
        
        Returns:
        - reciprocal_rank: The calculated reciprocal rank for the first true positive.
        - penalty_factor: A penalty factor based on false positives and missed positives.
        """
        # Track the reciprocal rank for the first true positive
        reciprocal_rank = 0
        found_true_labels = []
        predicted_positive_indices = []
        for rank, idx in enumerate(ranked_indices):
            if idx > rank:
                # Any index that moved to the top positions (initially, in the predicted positives)
                predicted_positive_indices.append(idx)

        false_positive_count = 0
        for rank, idx in enumerate(ranked_indices, start=1):
            if true_labels[idx] == 1:
                if reciprocal_rank == 0:
                    # Calculate the reciprocal rank for the first correct positive prediction
                    reciprocal_rank = 1.0 / rank
                found_true_labels.append(idx)
            elif idx in predicted_positive_indices and true_labels[idx] == 0:
                # Count false positives (predicted as positive but actually negative)
                false_positive_count += 1

        # Calculate missed true positives
        missed_positives = len([label for label in true_labels if label == 1]) - len(found_true_labels)

        # Calculate penalty
        # More false positives and missed positives result in a stronger penalty.
        # The penalty factor reduces with increasing false positives or missed positives.
        if false_positive_count + missed_positives> 0:
            penalty_factor = 1 / (1 + false_positive_count + missed_positives)
            reciprocal_rank *= penalty_factor

        return reciprocal_rank

    def retrieval(self, test, model, model_dir, output_file, num_neg, batch_size):
        eval_file = model_dir / output_file
        if eval_file.exists():
            eval_file.unlink()

        count = 0
        mrr_scores = []
        model.eval()
        print("start for retrieval")
        start_time = pd.Timestamp.now()

        # Create data loader to handle batch-wise evaluation
        data_loader = torch.utils.data.DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,  # Use the collate function defined earlier
        )

        with torch.no_grad():
            for batch in tqdm(data_loader):
                # Get the batch data
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"]  # Labels are not moved to GPU for retrieval purposes

                # Perform a forward pass
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                _, predicted_labels = output.logits  # Get similarities and predicted labels

                # Process each sample in the batch to compute MRR scores
                for idx in range(input_ids.size(0)):  # Iterate over the batch
                    sample_true_labels = labels[idx]  # Shape: [num_candidates]
                    sample_predicted_labels = predicted_labels[idx]  # Shape: [num_candidates]

                    indics_orig = list(range(num_neg+1))
                    pred_label = int(sample_predicted_labels.item()) 
                    indics_reranked = [pred_label] + [idx for idx in indics_orig if idx != pred_label]

                    # Calculate reciprocal rank for this sample
                    reciprocal_rank = self.calculate_mrr_with_penalty(indics_reranked, sample_true_labels.tolist())
                    mrr_scores.append(reciprocal_rank)

                    # Write individual results to file
                    data = {
                        'query': test[idx]["text_src"],
                        'candidates': test[idx]["candidates"],
                        'true_labels': sample_true_labels.tolist(),
                        'ranked_candidates': indics_reranked,
                        'mrr_score': reciprocal_rank,
                    }
                    with open(eval_file, 'a') as f:
                        f.write(json.dumps(data) + '\n')

                count += input_ids.size(0)

        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time_mean = inference_time.total_seconds() / count
        mean_mrr = np.mean(mrr_scores)

        return mean_mrr, count, inference_time_mean


    def matcher(self, test, model, model_dir, batch_size, output_file, threshold): 
        # Set the dataset to return PyTorch tensors in batches

        model.eval()
        all_predicted_labels = []
        all_labels = []
        predictions_data = []
        print("start for matching")
        start_time = pd.Timestamp.now()
        # Iterate over the dataset in batches
        data_loader = torch.utils.data.DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,  # Use the collate function defined earlier
        )
        with torch.no_grad():
            for batch in tqdm(data_loader):               
                # Move each tensor in the batch to the model's device
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"]  # Labels are not moved to GPU for retrieval purposes

                # Perform a forward pass
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                predicted_labels, _ = output.logits  # Get similarities and predicted labels
                all_labels.append(labels.view(-1).cpu())  # Ensure to flatten each batch's labels
                all_predicted_labels.append(predicted_labels.view(-1).cpu())  # Ensure to flatten each batch's labels

                # Prepare data for CSV file
                labels_np = labels.cpu().numpy().flatten()
                predicted_labels = predicted_labels.cpu().numpy().flatten()
                for idx in range(len(labels_np)):
                    predictions_data.append({
                        'query': test["text_src"][idx],  # Assuming each batch starts with the query at index i
                        'candidate': test["text_tgt"][idx],  # Assuming similar structure; customize if necessary
                        'predicted_label': predicted_labels[idx],
                        'true_label': labels_np[idx],
                    })

        # Concatenate all probabilities and labels across batches
        all_labels = torch.cat(all_labels, dim=0)
        all_predicted_labels = torch.cat(all_predicted_labels, dim=0)
        # Calculate metrics
        f1 = f1_score(all_labels.numpy(), all_predicted_labels.numpy())
        recall = recall_score(all_labels.numpy(), all_predicted_labels.numpy())
        precision = precision_score(all_labels.numpy(), all_predicted_labels.numpy())

        count = len(all_labels)
        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time_mean = inference_time.total_seconds() / count

        # Save predictions to CSV
        predictions_df = pd.DataFrame(predictions_data)
        eval_pred_path = os.path.join(model_dir, 'eval_pred.csv')
        predictions_df.to_csv(eval_pred_path, index=False)
        print(f"F1 Score: {f1}, Recall: {recall}, Precision: {precision}")
        return f1, recall, precision, count, inference_time_mean


    def evaluate(self, test, model_dir, output_file, model, tokenizer, max_length, num_neg, threshold, batch_size=32, flag_fine_tuning=True, eval_type='retrieval'):
        print(str(model_dir))
        if flag_fine_tuning:
            model, tokenizer = self.merge_model(model, model_dir, eval_type)

        if eval_type=='retrieval':
            mean_mrr, count, inference_time_mean = self.retrieval(test, model, model_dir, output_file, num_neg, batch_size)
        
            report = {
                'num_query': count,
                'inference_time_per_query': inference_time_mean,
                'mean_mrr': mean_mrr,
            }

            with open(model_dir / "inference_report_retrieval.json", 'w') as f:
                f.write(json.dumps(report))

            print(f"test MRR@10: {mean_mrr}")
            print(f"inference_time_per_query: {inference_time_mean}")
        
        elif eval_type=='pairwise':
            f1, recall, precision, count, inference_time_mean = self.matcher(test, model, model_dir, batch_size, output_file, threshold)

            report = {
                'num_pair': count,
                'inference_time_per_query': inference_time_mean,
                'f1': f1,
                'recall': recall,
                'precision': precision
            }

            with open(model_dir / "inference_report_pairwise.json", 'w') as f:
                f.write(json.dumps(report))

            print(f"f1: {f1}")
            print(f"recall: {recall}")
            print(f"precision: {precision}")
            print(f"inference_time_per_query: {inference_time_mean}")
