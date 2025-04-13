from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer
from peft import  AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM
from torch import nn





class Inferencer:
    def __init__(self) -> None:
        '''
        params:task_name: str: name of the task
        params:train_type: str: type of the training data in ['train','train_small']
        params:val_type: str: type of the validation data in ['test']
        params:test_type: str: type of the test data in ['test','test_big']
        '''
        print('Using the model for inference...')
        device = torch.device("cuda:0")
       

    def merge_model(self, finetuned_model_dir:Path, labels, label2id, id2label, emb_type=None, input_type=None, num_cls_layers=1):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        compute_dtype = getattr(torch, "float16")   
        print('merge model emb_type', emb_type)
        model =  AutoPeftModelForSequenceClassification.from_pretrained(
                        str(finetuned_model_dir)+'/',
                        torch_dtype=compute_dtype,
                        return_dict=False,
                        low_cpu_mem_usage=True,
                        device_map='auto',
                        num_labels = len(labels),
                    )
        model.to('cuda:0')
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print('merge_model, 1model', model)
        '''
        print('model.config.id2label', model.config.id2label)
        print('model.config.label2id', model.config.label2id)
        print('!!!!!!!!!!!!!!!!!!!!!!!', id2label)
        print('!!!!!!!!!!!!!!!!!!!!!!!', label2id)
        '''
        model.config.id2label = id2label
        model.config.label2id = label2id
        '''
        print('model.config.id2label', model.config.id2label)
        print('model.config.label2id', model.config.label2id)
        print('model.score.weight', model.score.weight)
        '''


        model = model.merge_and_unload()
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print('model merged')
        print('merge_model, 2model', model)
        model.to('cuda:0')
        model.config.pad_token_id = tokenizer.pad_token_id
       
        print('merge_model, 4model', model)
        
        print('merge_model, 4model.config.id2label', model.config.id2label)
        print('merge_model, 4model.config.label2id', model.config.label2id)
        print('merge_model, 4model.score.weight', model.score.weight)

        if 'mistral' in str(finetuned_model_dir):
            model.config.sliding_window = 4096
        
        return model, tokenizer

    
    
    def predict(self, test, model, tokenizer, labels, label2id, id2label, output_dir, output_file):
        #test is a huggingface dataset
        # get length of the dataset
    
        
        eval_file = output_dir / output_file
        print('eval_file', eval_file)
        if eval_file.exists():
            eval_file.unlink()
        
        for i in tqdm(range(len(test))):
             
            #print('############', test[i]["text"])
            #print('1. prompt', prompt)
            inputs = tokenizer(test[i]["text"], return_tensors="pt").to('cuda:0')
            with torch.no_grad():
                #print('model(**inputs)',model(**inputs, return_dict=True))
                logits = model(**inputs, return_dict=True).logits.to('cuda:0')
                #print('logits', logits)
            predicted_class_id = logits.argmax().item()
            pred = id2label[int(predicted_class_id)]

            if 'ei' in test.column_names:
                true = test[i]['ei']
                data = {'doc_name':[test[i]['doc_name']], 
                    'node_ix_src':[test[i]['node_ix_src']], 
                    'node_ix_tgt':[test[i]['node_ix_tgt']], 
                    'text_src':[test[i]['text_src']], 
                    'text_tgt':[test[i]['text_tgt']],
                    'ea':[test[i]['ea']],
                    'ei':[pred],
                    'true':[true]}
            else:
                data = {'doc_name':[test[i]['doc_name']], 
                        'node_ix_src':[test[i]['node_ix_src']], 
                        'node_ix_tgt':[test[i]['node_ix_tgt']], 
                        'text_src':[test[i]['text_src']], 
                        'text_tgt':[test[i]['text_tgt']],
                        'ea':[test[i]['ea']],
                        'ei':[pred]}
            

            a = pd.DataFrame(data)

            #a = pd.DataFrame({ "true":[id2label[test[i]['label']]], "pred":[pred]})
            a.to_csv(eval_file,mode="a",index=False,header=not eval_file.exists())
        

    def run_inference(self, test, labels, label2id, id2label, model_dir, output_dir=None, output_file=None, do_predict = True, model=None,
                 tokenizer=None, emb_type=None, input_type=None, num_cls_layers=1):
        """
        Evaluate the model using accuracy, classification report, and confusion matrix
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param labels2id: Dictionary mapping labels to ids
        """
        print('run_inference emb_type', emb_type)
        print('load model', model_dir)
        if model is None or tokenizer is None:
            model, tokenizer = self.merge_model(model_dir, labels, label2id, id2label, emb_type=emb_type, input_type=input_type, num_cls_layers=num_cls_layers)

        start_time = pd.Timestamp.now()
        if do_predict:
            if output_dir is None:
                output_dir = model_dir
            self.predict(test, model, tokenizer, labels, label2id, id2label, output_dir, output_file)
        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time = inference_time.total_seconds()
        with open (output_dir / f"{Path(output_file).stem}_inference_time.json", 'w') as f:
            json.dump({'inference_time':int(inference_time)}, f, indent=4)


        '''
         
        df = pd.read_csv(output_dir / "eval_pred.csv")
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

        eval_file = output_dir / "eval_report.json"
        if eval_file.exists():
            eval_file.unlink()
        with open(str(eval_file), 'w') as f:
            json.dump(class_report, f, indent=4)
        eval_file = output_dir / "eval_cm.csv"
        if eval_file.exists():
            eval_file.unlink()
        df = pd.DataFrame(conf_matrix, columns=labels, index=labels)
        print('\nConfusion Matrix:')
        print(df)
        df.to_csv(eval_file)
        '''

def main():
    ''

if __name__ == "__main__":
    main()