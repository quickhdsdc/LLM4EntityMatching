import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Evaluater:
    def __init__(self) -> None:
        pass

    def predict(self, test, model, labels, output_dir, eval_type, model_root_path):
        # test is a huggingface dataset
        # get length of the dataset
        eval_file = output_dir / f"eval_pred_{eval_type}.csv"
        print('eval_file', eval_file)
        if eval_file.exists():
            eval_file.unlink()

        for i in tqdm(range(len(test))):
            instruction = test[i]["instr"]
            input = test[i]["input"]
            # print('1. prompt', prompt)
            response = model.chat.completions.create(
                model=model_root_path,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input}
                ],
                temperature=0.1
            )
            try:
                answer = response.choices[0].message.content
                found = False
                for l in labels:
                    if l.lower() in answer.lower():
                        pred = l
                        found = True
                        break
                if not found:
                    pred = "none"
            except Exception as e:
                print(f"Prediction failed with error: {e}")
                pred = "none"
            if test[i]['label'] == 0:
                true = 'no'
            elif test[i]['label'] == 1:
                true = 'yes'
            else:
                true = test[i]['label']
            a = pd.DataFrame({"true": [true], "pred": [pred], "answer": [answer], "instr": [instruction],
                              "input": [input]})
            a.to_csv(eval_file, mode="a", index=False, header=not eval_file.exists())


    def calculate_mrr_with_penalty(self, ranked_indices, true_labels, predicted_labels):
        """
        Calculate MRR with penalty for false positives and missed detections using true_labels and predicted_labels.

        Args:
        - ranked_indices: The list of ranked candidate indices after prediction.
        - true_labels: The ground truth labels (list of 0s and 1s).
        - predicted_labels: The predicted labels (list of 0s and 1s).

        Returns:
        - reciprocal_rank: The calculated reciprocal rank for the first true positive.
        - penalty_factor: A penalty factor based on false positives and missed positives.
        """
        # Track the reciprocal rank for the first true positive
        reciprocal_rank = 0
        found_true_labels = []

        for rank, idx in enumerate(ranked_indices, start=1):
            if true_labels[idx] == 1:
                if reciprocal_rank == 0:
                    # Calculate the reciprocal rank for the first correct positive prediction
                    reciprocal_rank = 1.0 / rank
                found_true_labels.append(idx)

        # Calculate false positives and missed positives directly
        false_positive_count = sum(
            1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1
        )
        missed_positives = sum(
            1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0
        )

        # Calculate penalty
        # More false positives and missed positives result in a stronger penalty.
        if false_positive_count + missed_positives > 0:
            penalty_factor = 1 / (1 + false_positive_count + missed_positives)
            reciprocal_rank *= penalty_factor

        return reciprocal_rank

    def retrieval(self, test, model, id2label, output_dir, model_root_path):
        # Load and initialize parameters
        eval_file = output_dir / "eval_retrieval.csv"
        if eval_file.exists():
            eval_file.unlink()

        retrieval_results = []
        retrieval_scores = []

        # Iterate over the dataset
        for i in tqdm(range(0, len(test), 10)):
            # Create a batch for the 10 candidates of the same query
            inputs_text = test[i: i + 10]

            # Initialize lists for predictions and true labels
            predicted_labels = []
            true_labels = []

            for j in range(10):
                instruction = inputs_text["instr"][j]
                input = inputs_text["input"][j]
                try:
                    response = model.chat.completions.create(
                        model=model_root_path,
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": input}
                        ],
                        temperature=0.1
                    )
                    answer = response.choices[0].message.content
                    found = False
                    for l in id2label.values():
                        if l.lower() in answer.lower():
                            predicted_label = 1 if l.lower() == 'yes' else 0
                            found = True
                            break
                    if not found:
                        predicted_label = 0  # Default to 'no' if no label found
                except Exception as e:
                    print(f"Prediction failed with error: {e}")
                    predicted_label = 'none'

                # Append the predicted label to the list
                predicted_labels.append(predicted_label)
                label = inputs_text['label'][j]
                true_labels.append(label)

            # Re-rank candidates based on predicted labels
            positive_indices = [idx for idx, label in enumerate(predicted_labels) if label == 1]
            negative_indices = [idx for idx, label in enumerate(predicted_labels) if label == 0]

            # Combine positive and negative indices to get the new ranking order
            reranked_indices = positive_indices + negative_indices

            # Extract the true labels for the current query's candidates
            reciprocal_rank = self.calculate_mrr_with_penalty(reranked_indices, true_labels, predicted_labels)
            print(f"reciprocal_rank: {reciprocal_rank}")

            # Add the reciprocal rank to the scores list for calculating the mean later
            retrieval_scores.append(reciprocal_rank)

            # Optional: save prediction results for inspection
            results_data = {
                'true_labels': true_labels,
                'predicted_labels': ['yes' if pred == 1 else 'no' for pred in predicted_labels],
                'reranked_indices': reranked_indices,
                'mrr': reciprocal_rank
            }
            retrieval_results.append(results_data)
            pd.DataFrame([results_data]).to_csv(eval_file, mode="a", index=False, header=not eval_file.exists())

        # Calculate the mean MRR@10 over all queries
        mean_mrr = np.mean(retrieval_scores)
        print(f"Mean MRR@10: {mean_mrr:.3f}")

        # Save overall MRR score
        with open(output_dir / "mrr_score.json", 'w') as f:
            json.dump({'mean_mrr': mean_mrr}, f, indent=4)


    def evaluate(self, test, labels, label2id, id2label, model_dir, model, tokenizer=None, max_length=None, eval_type = 'pairwise_old', flag_fine_tuning=True):
        start_time = pd.Timestamp.now()
        if eval_type == 'retrieval':
            self.retrieval(test, model, id2label, model_dir, "gpt-4o-2024-05-13") # gpt-4-1106-preview, gpt-4o-2024-05-13
        else:
            self.predict(test, model, labels, model_dir, eval_type, "gpt-4o-2024-05-13")
        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time = inference_time.total_seconds()
        with open(model_dir / "inference_time.json", 'w') as f:
            json.dump({'inference_time': int(inference_time)}, f, indent=4)

        if 'pairwise' in eval_type:
            df = pd.read_csv(model_dir / f"eval_pred_{eval_type}.csv")
            none_nr = len(df[df['pred'] == 'none'])
            df = df[df['pred'] != 'none']
            y_pred = df["pred"]
            y_true = df["true"]

            # Map labels to ids
            label2id['none'] = -1
            map_func = lambda label: label2id[label]
            y_true = np.vectorize(map_func)(y_true)
            y_pred = np.vectorize(map_func)(y_pred)

            # Calculate accuracy
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            print(f'Accuracy: {accuracy:.3f}')

            # Generate accuracy report
            class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels, output_dict=True,
                                                 zero_division=0)
            print('\nClassification Report:')
            class_report['none_nr'] = none_nr
            print(class_report)

            # Generate confusion matrix
            y_true_labels = [labels[i] for i in y_true]
            y_pred_labels = [labels[i] for i in y_pred]
            conf_matrix = confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels, labels=labels)
            # print('\nConfusion Matrix:')
            # print(conf_matrix)

            eval_file = model_dir / f"eval_report_{eval_type}.json"
            with open(str(eval_file), 'w') as f:
                json.dump(class_report, f, indent=4)
            eval_file = model_dir / f"eval_cm_{eval_type}.csv"
            df = pd.DataFrame(conf_matrix, columns=labels, index=labels)
            print('\nConfusion Matrix:')
            print(df)
            df.to_csv(eval_file)

