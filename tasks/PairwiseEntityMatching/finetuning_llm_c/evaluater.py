from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
from .modelling_llama import LlamaForSequenceClassification, MistralForSequenceClassification
from .modelling_peft import AutoPeftModelForSequenceClassificationCustom

class Evaluater:
    def __init__(self) -> None:

        print('Evaluating the model...')

    def merge_model(self, finetuned_model_dir:Path, labels, label2id, id2label, pooling_type):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        compute_dtype = getattr(torch, "float16")   
        model =  AutoPeftModelForSequenceClassificationCustom.from_pretrained(
                        str(finetuned_model_dir)+'/',
                        torch_dtype=compute_dtype,
                        return_dict=False,
                        low_cpu_mem_usage=True,
                        device_map='auto',
                        num_labels = len(labels),
                    )
        model.config.id2label = id2label
        model.config.label2id = label2id
        model = model.merge_and_unload()
        score = model.score
        model = model.model
        if 'Mistral' in str(finetuned_model_dir) or 'mistral' in str(finetuned_model_dir):
            model = MistralForSequenceClassification(model.config, model=model, score=score, pooling_type=pooling_type)
            model.config.sliding_window = 4096
            print('model has been merged')
        elif 'lama' in str(finetuned_model_dir):
            model = LlamaForSequenceClassification(model.config, model=model, score=score, pooling_type=pooling_type)
            print('model has been merged')
        model.config.pad_token_id = tokenizer.pad_token_id        

        return model, tokenizer

    def predict(self, test, model, tokenizer, max_length, id2label, output_dir, eval_type):
        #test is a huggingface dataset
        # get length of the dataset
        eval_file = output_dir / f"eval_pred_{eval_type}.csv"
        print('eval_file', eval_file)
        if eval_file.exists():
            eval_file.unlink()
        model.eval()
        for i in tqdm(range(len(test))):
            inputs = tokenizer(test[i]["text"], max_length=max_length, padding="max_length", truncation=True,return_tensors="pt")
            inputs = inputs.to(next(model.parameters()).device)
            with torch.no_grad():
                logits = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"], return_dict=True).logits
            predicted_class_id = logits.argmax().item()
            pred = id2label[int(predicted_class_id)]

            a = pd.DataFrame({ "true":[id2label[test[i]['label']]], "pred":[pred]})
            a.to_csv(eval_file,mode="a",index=False,header=not eval_file.exists())
    
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
        
    def retrieval(self, test, model, tokenizer, max_length, id2label, output_dir):
        # Load and initialize parameters
        eval_file = output_dir / "eval_retrieval.csv"
        if eval_file.exists():
            eval_file.unlink()
        
        model.eval()
        retrieval_results = []
        retrieval_scores = []

        # Iterate over the dataset
        with torch.no_grad():
            for i in tqdm(range(0, len(test), 10)):
                # Create a batch for the 10 candidates of the same query
                inputs_text = test[i: i + 10]
                inputs = tokenizer(
                    [inputs_text["text"][j] for j in range(10)], 
                    max_length=max_length, 
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(next(model.parameters()).device)

                # Get predictions
                logits = model(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    return_dict=True
                ).logits
                # Predicted labels using argmax to determine whether class is 0 or 1
                predicted_labels = logits.argmax(dim=-1).cpu().numpy()

                # Original candidate indices
                original_indices = list(range(10))

                # Re-rank candidates based on predicted labels
                # If predicted label == 1, move candidate to the top
                positive_indices = [idx for idx, label in enumerate(predicted_labels) if label == 1]
                negative_indices = [idx for idx, label in enumerate(predicted_labels) if label == 0]

                # Combine positive and negative indices to get the new ranking order
                reranked_indices = positive_indices + negative_indices

                # Extract the true labels for the current query's candidates
                true_labels = [inputs_text['label'][j] for j in original_indices]

                reciprocal_rank = self.calculate_mrr_with_penalty(reranked_indices, true_labels, predicted_labels)

                # Add the reciprocal rank to the scores list for calculating the mean later
                retrieval_scores.append(reciprocal_rank)

                # Optional: save prediction results for inspection
                results_data = {
                    'query_id': test[i]["id_left"],
                    'ranked_candidates': [inputs_text['id_right'][idx] for idx in reranked_indices],
                    'true_labels': true_labels,
                    'predicted_labels': predicted_labels.tolist(),
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


    def evaluate(self, test, labels, label2id, id2label, model_dir, max_length=512, output_dir=None, model=None,
                 tokenizer=None, pooling_type='orig', eval_type='pairwise_new', flag_fine_tuning=True):
        """
        Evaluate the model using accuracy, classification report, and confusion matrix
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param label2id: Dictionary mapping labels to ids
        """
        if flag_fine_tuning:
            model, tokenizer = self.merge_model(model_dir, labels, label2id, id2label,pooling_type=pooling_type)

        if output_dir is None:
            output_dir = Path(model_dir)

        start_time = pd.Timestamp.now()
        if eval_type == 'retrieval':
            self.retrieval(test, model, tokenizer, max_length, id2label, output_dir)
        else:
            self.predict(test, model, tokenizer, max_length, id2label, output_dir, eval_type)
        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time = inference_time.total_seconds()
        with open (output_dir / "inference_time.json", 'w') as f:
            json.dump({'inference_time':int(inference_time)}, f, indent=4)
       
        if 'pairwise' in eval_type: 
            df = pd.read_csv(output_dir / f"eval_pred_{eval_type}.csv")
            none_nr = len(df[df['pred'] == 'none'])
            df = df[df['pred'] != 'none']
            y_pred = df["pred"]
            y_true = df["true"]
            print(df)
            
            # Map labels to ids
            #label2id['none'] = -1
            map_func = lambda label: label2id[label]
            y_true = np.vectorize(map_func)(y_true)
            y_pred = np.vectorize(map_func)(y_pred)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            print(f'Accuracy: {accuracy:.3f}')
            
            # Generate accuracy report
            class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels, output_dict=True, zero_division=0)
            print('\nClassification Report:')
            class_report['none_nr'] = none_nr
            print(class_report)

            # Generate confusion matrix
            y_true_labels = [labels[i] for i in y_true]
            y_pred_labels = [labels[i] for i in y_pred]
            conf_matrix = confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels, labels=labels)
            #print('\nConfusion Matrix:')
            #print(conf_matrix)

            eval_file = output_dir / f"eval_report_{eval_type}.json"
            if eval_file.exists():
                eval_file.unlink()
            with open(str(eval_file), 'w') as f:
                json.dump(class_report, f, indent=4)
            eval_file = output_dir / f"eval_cm_{eval_type}.csv"
            if eval_file.exists():
                eval_file.unlink()
            df = pd.DataFrame(conf_matrix, columns=labels, index=labels)
            print('\nConfusion Matrix:')
            print(df)
            df.to_csv(eval_file)
        

def main():
    ''

if __name__ == "__main__":
    main()