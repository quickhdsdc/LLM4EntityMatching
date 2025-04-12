from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from peft import PeftModelForFeatureExtraction
from transformers import AutoTokenizer
import torch.nn.functional as F
from .modelling_plm import EntityRetrieverMPNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import os

class Evaluater:
    def __init__(self) -> None:
        print('Evaluating the model...')

    def load_model(self, finetuned_model_dir: Path):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = EntityRetrieverMPNet.from_pretrained(str(finetuned_model_dir), device_map="cuda" if torch.cuda.is_available() else "cpu")
        return model, tokenizer

    def prepare_inference_input(self, sample, model, num_neg, used_indices):
        device = next(model.parameters()).device

        input_ids_query = sample["input_ids_query"]
        attention_mask_query = sample["attention_mask_query"]
        input_ids_candidates = sample["input_ids_candidates"]
        attention_mask_candidates = sample["attention_mask_candidates"]

        input_ids_query = torch.tensor(input_ids_query).to(device)
        attention_mask_query = torch.tensor(attention_mask_query).to(device)

        # Get remaining candidates
        remaining_indices = [i for i in range(len(input_ids_candidates)) if i not in used_indices]

        # Select num_neg + 1 candidates randomly
        if len(remaining_indices) < num_neg + 1:
            needed = num_neg + 1 - len(remaining_indices)
            remaining_indices += random.sample(used_indices, needed)

        selected_indices = random.sample(remaining_indices, num_neg + 1)
        selected_candidates = [(input_ids_candidates[i], attention_mask_candidates[i]) for i in selected_indices]
        input_ids_candidates, attention_mask_candidates = zip(*selected_candidates)

        # Convert candidates to tensors and move to device
        input_ids_candidates = [torch.tensor(c).unsqueeze(0).to(device) for c in input_ids_candidates]
        input_ids_candidates = torch.cat(input_ids_candidates, dim=0).unsqueeze(0)  # Add batch dimension
        attention_mask_candidates = [torch.tensor(c).unsqueeze(0).to(device) for c in attention_mask_candidates]
        attention_mask_candidates = torch.cat(attention_mask_candidates, dim=0).unsqueeze(0)

        return {
            "input_ids_query": input_ids_query,
            "attention_mask_query": attention_mask_query,
            "input_ids_candidates": input_ids_candidates,
            "attention_mask_candidates": attention_mask_candidates,
            "selected_indices": selected_indices
        }

    def calculate_mrr_at_10(self, ranked_list, true_label):
        try:
            rank_index = ranked_list.index(true_label)
            return 1 / (rank_index + 1) if rank_index < 10 else 0
        except ValueError:
            return 0
    
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

    def retrieval(self, test, model, model_dir, output_file, num_neg):
        eval_file = model_dir / output_file
        if eval_file.exists():
            eval_file.unlink()

        count = 0
        mrr_scores = []
        model.eval()
        print("start for retrieval")
        start_time = pd.Timestamp.now()

        for i in tqdm(range(len(test))):
            # true_label = np.argmax(test[i]["labels"])
            used_indices = []
            candidates = list(range(len(test[i]['input_ids_candidates'])))
            candidates_orig = candidates
            reranked_candidates = []

            while candidates:
                prepared_input = self.prepare_inference_input(test[i], model, num_neg, used_indices)
                input_ids_query = prepared_input["input_ids_query"]
                attention_mask_query = prepared_input["attention_mask_query"]
                input_ids_candidates = prepared_input["input_ids_candidates"]
                attention_mask_candidates = prepared_input["attention_mask_candidates"]
                selected_indices = prepared_input["selected_indices"]

                with torch.no_grad():
                    prediction = model(input_ids=input_ids_query, 
                                       attention_mask=attention_mask_query, 
                                       input_ids_candidates=input_ids_candidates, 
                                       attention_mask_candidates=attention_mask_candidates).logits

                pred_label = prediction[1].detach().cpu().numpy()[0]  # predicted candidate index
                absolute_pred_label = selected_indices[pred_label]
                # reranking
                reranked_candidates = [absolute_pred_label] + [idx for idx in candidates_orig if idx != absolute_pred_label]

                # Add all selected indices to used_indices except the predicted one
                used_indices.extend([idx for idx in selected_indices if idx != absolute_pred_label])
                # Keep the predicted candidate for the next iteration
                candidates = [c for c in candidates if c not in used_indices]

                if len(candidates) == 1:
                    break
            true_labels = test[i]['labels']
            reciprocal_rank = self.calculate_mrr_with_penalty(reranked_candidates, true_labels)
            # mrr_scores.append(self.calculate_mrr_at_10(reranked_candidates, true_labels))
            mrr_scores.append(reciprocal_rank)

            count += 1
            data = {
                'query': test[i]["text"],
                'candidates': test[i]["candidates"],
                'true_labels': test[i]["labels"],
                'ranked_candidates': reranked_candidates,
                'mrr_score': mrr_scores[-1],
            }

            with open(eval_file, 'a') as f:
                f.write(json.dumps(data) + '\n')

        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time_mean = inference_time.total_seconds() / count
        mean_mrr = np.mean(mrr_scores)

        return mean_mrr, count, inference_time_mean


    def matcher(self, test, model, model_dir, batch_size, output_file, threshold): 
        # Set the dataset to return PyTorch tensors in batches
        test.set_format(type='torch', columns=['input_ids_query', 'attention_mask_query', "text_src", "text_tgt",
                                               'input_ids_candidate', 'attention_mask_candidate', 'label'])

        model.eval()
        all_probabilities = []
        all_labels = []
        predictions_data = []
        print("start for matching")
        start_time = pd.Timestamp.now()
        # Iterate over the dataset in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(test), batch_size)):
                batch = test[i: i + batch_size]
                
                # Move each tensor in the batch to the model's device
                query_input_ids = batch['input_ids_query'].squeeze(1).to(model.device)
                query_attention_mask = batch['attention_mask_query'].squeeze(1).to(model.device)
                candidate_input_ids = batch['input_ids_candidate'].squeeze(1).to(model.device)
                candidate_attention_mask = batch['attention_mask_candidate'].squeeze(1).to(model.device)
                labels = batch['label'].to(model.device)

                # Forward pass to get logits (similarities)
                similarities = model(
                    input_ids=query_input_ids, 
                    attention_mask=query_attention_mask, 
                    input_ids_candidates=candidate_input_ids, 
                    attention_mask_candidates=candidate_attention_mask
                )

                # Apply sigmoid to get probabilities
                # probabilities = torch.sigmoid(similarities)
                # probabilities = (similarities + 1) / 2
                probabilities = torch.diag(similarities)

                # Collect probabilities and labels for threshold optimization
                all_probabilities.append(probabilities.view(-1).cpu())  # Ensure to flatten each batch's probabilities
                all_labels.append(labels.view(-1).cpu())  # Ensure to flatten each batch's labels

                # Prepare data for CSV file
                probabilities_np = probabilities.cpu().numpy().flatten()
                labels_np = labels.cpu().numpy().flatten()
                predicted_labels = (probabilities_np >= threshold).astype(int)
                 # Store each row's data for CSV (query, candidate, predicted label, true label)
                for idx in range(len(labels_np)):
                    predictions_data.append({
                        'query': batch["text_src"][idx],  # Assuming each batch starts with the query at index i
                        'candidate': batch["text_tgt"][idx],  # Assuming similar structure; customize if necessary
                        'probabilities': probabilities[idx].cpu().numpy(),
                        'predicted_label': predicted_labels[idx],
                        'true_label': labels_np[idx],
                    })

        # Concatenate all probabilities and labels across batches
        all_probabilities = torch.cat(all_probabilities, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Threshold-based prediction
        predicted_labels = (all_probabilities > threshold).long()

        # Calculate metrics
        f1 = f1_score(all_labels.numpy(), predicted_labels.numpy())
        recall = recall_score(all_labels.numpy(), predicted_labels.numpy())
        precision = precision_score(all_labels.numpy(), predicted_labels.numpy())

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
            model, tokenizer = self.load_model(model_dir)

        if eval_type=='retrieval':
            mean_mrr, count, inference_time_mean = self.retrieval(test, model, model_dir, output_file, num_neg)
        
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
