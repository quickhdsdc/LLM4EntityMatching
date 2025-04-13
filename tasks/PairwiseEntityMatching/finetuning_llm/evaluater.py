from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .data_preprocessor import RESPONSE_KEY
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
from peft import AutoPeftModelForCausalLM

class Evaluater:
    def __init__(self) -> None:
        '''
        params:task_name: str: name of the task
        params:train_type: str: type of the training data in ['train','train_small']
        params:val_type: str: type of the validation data in ['test']
        params:test_type: str: type of the test data in ['test','test_big']
        '''
        print('Evaluating the model...')


    def merge_model(self, finetuned_model_dir:Path, labels, label2id, id2label):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        compute_dtype = getattr(torch, "float16")   
        model = AutoPeftModelForCausalLM.from_pretrained(
                    str(finetuned_model_dir)+'/',
                    torch_dtype=compute_dtype,
                    return_dict=False,
                    low_cpu_mem_usage=True,
                    device_map='auto',
                )

        model = model.merge_and_unload()
        model.config.pad_token_id = tokenizer.pad_token_id

        if 'mistral' in str(finetuned_model_dir):
            model.config.sliding_window = 4096
       
        # print('merge_model, 4model', model)
        
        return model, tokenizer


    def predict(self, test, model, tokenizer, labels, output_dir, response_key, eval_type):
        #test is a huggingface dataset
        # get length of the dataset
        eval_file = output_dir / f"eval_pred_{eval_type}.csv"
        print('eval_file', eval_file)
        if eval_file.exists():
            eval_file.unlink()
        
        for i in tqdm(range(len(test))):
            prompt = test[i]["text"]
            #print('1. prompt', prompt)
            pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 10, 
                        temperature = 0.1,
                       )
            try:
                result = pipe(prompt)
                answer = result[0]['generated_text'].split(response_key)[-1]
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
                # If the prediction fails, generate a random label from the available labels
                pred = "none"  
            if test[i]['label'] == 0:
                true = 'no'
            elif test[i]['label'] == 1:
                true = 'yes'          
            else:
                true = test[i]['label']
            a = pd.DataFrame({ "true":true, "pred":[pred], "answer":[answer], "prompt":[prompt]})
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
        
    def retrieval(self, test, model, tokenizer, response_key, id2label, output_dir):
        from transformers import pipeline

        # Load and initialize parameters
        eval_file = output_dir / "eval_retrieval.csv"
        if eval_file.exists():
            eval_file.unlink()

        model.eval()
        retrieval_results = []
        retrieval_scores = []

        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=10, 
                        temperature=0.1)

        # Iterate over the dataset
        with torch.no_grad():
            for i in tqdm(range(0, len(test), 10)):
                # Create a batch for the 10 candidates of the same query
                inputs_text = test[i: i + 10]
                
                # Initialize lists for predictions and true labels
                predicted_labels = []
                true_labels = []

                for j in range(10):
                    prompt = inputs_text["text"][j]

                    try:
                        # Use the text-generation pipeline to get predictions
                        result = pipe(prompt)
                        answer = result[0]['generated_text'].split(response_key)[-1]
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
                        predicted_label = 0

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

    def evaluate(self, test, labels, label2id, id2label, output_dir, response_key, eval_type = 'pairwise_old', model=None,
                 tokenizer=None, flag_fine_tuning=True):
        """
        Evaluate the model using accuracy, classification report, and confusion matrix
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param labels2id: Dictionary mapping labels to ids
        """
        if flag_fine_tuning:
            model, tokenizer = self.merge_model(output_dir, labels, label2id, id2label)
        
        start_time = pd.Timestamp.now()
        if eval_type == 'retrieval':
            self.retrieval(test, model, tokenizer, response_key, id2label, output_dir)
        else:
            self.predict(test, model, tokenizer, labels, output_dir, response_key, eval_type)
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
            
            # Map labels to ids
            label2id['none'] = -1
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
            with open(str(eval_file), 'w') as f:
                json.dump(class_report, f, indent=4)
            eval_file = output_dir / f"eval_cm_{eval_type}.csv"
            df = pd.DataFrame(conf_matrix, columns=labels, index=labels)
            print('\nConfusion Matrix:')
            print(df)
            df.to_csv(eval_file)
        
