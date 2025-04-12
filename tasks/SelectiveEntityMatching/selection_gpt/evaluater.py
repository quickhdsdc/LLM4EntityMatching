import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import faiss
import pickle
import os
import random
import re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Evaluater:
    def __init__(self) -> None:
        pass


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
        if false_positive_count + missed_positives > 0:
            penalty_factor = 1 / (1 + false_positive_count + missed_positives)
            reciprocal_rank *= penalty_factor

        return reciprocal_rank

    def retrieval(self, test, model, model_dir, output_file):
        eval_file = model_dir / output_file
        if eval_file.exists():
            eval_file.unlink()

        count = 0
        mrr_scores = []
        print("start for retrieval")
        start_time = pd.Timestamp.now()

        for i in tqdm(range(len(test))):
            instruction = test[i]['instr']
            query = test[i]['text_src']
            candidates = test[i]['candidates']
            candidates = candidates.strip("[]").split("', '")
            candidates = [c.strip("'") for c in candidates]
            true_labels = test[i]['labels']
            text_candidates = [f"candidate{i+1}: {c}" for i, c in enumerate(candidates)]
            rank_orig = list(range(len(candidates)))

            response = model.chat.completions.create(
                model="gpt-4-1106-preview", # gpt-4-1106-preview, o1-preview-2024-09-12
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"query: {query}\ncandidates: {text_candidates}"}
                ],
                temperature=0.1
            )
            result = response.choices[0].message.content
            try:
                absolute_pred_label = int(re.search(r'\b\d+\b', result).group())
            except AttributeError:
                absolute_pred_label = len(candidates)-1
            if absolute_pred_label > len(candidates)-1:
                absolute_pred_label = len(candidates) - 1
            reranked_candidates = [absolute_pred_label] + [idx for idx in rank_orig if idx != absolute_pred_label]
            reciprocal_rank = self.calculate_mrr_with_penalty(reranked_candidates, true_labels)
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


    def evaluate(self, test, model_dir, output_file=None, model=None, tokenizer=None, max_length=None, flag_fine_tuning=None, num_neg=None, batch_size=None, threshold=None, eval_type=None):
        mean_mrr, count, inference_time_mean = self.retrieval(test, model, model_dir, output_file)

        report = {
            'num_query': count,
            'inference_time_per_query': inference_time_mean,
            'mean_mrr': mean_mrr,
        }

        with open(model_dir / "inference_report_retrieval.json", 'w') as f:
            f.write(json.dumps(report))

        print(f"test MRR@10: {mean_mrr}")
        print(f"inference_time_per_query: {inference_time_mean}")

