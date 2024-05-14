
import numpy as np
import json
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from evaluate import load

def metrics_eval(df):
    bleu_avg=0.
    bert_avg = 0.
    acc = 0.

    bert_score = load("bertscore")

    for i in range(len(df['answers'])):
        if df['predict'][i].lower()==df['answers'][i].lower(): 
            acc+=1

        reference = str(df['answers'][i])
        candidate = df['predict'][i]

        # chencherry = SmoothingFunction()
        bleu_1 = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))
        bleu_avg+=bleu_1

        a = bert_score.compute(references =[reference],predictions =[candidate],model_type = 'bert-base-uncased')
        bert_avg+= a['f1'][0]

    return round(bleu_avg/len(df['answers']),3), round(bert_avg/len(df['answers']),3), round(acc/len(df['answers']),3)

def evaluate(result_dir):
    df = pd.read_csv(result_dir)

    metrics = {}

    for type in ['what', 'where', 'when', 'why', 'who', 'how']:
        sub_df = df[df['types'] == type].reset_index(drop=True)

        bleu, bert, acc = metrics_eval(sub_df)

        metrics[type] = [bleu, bert, acc]

        print(f'{type}', metrics[type])
    
    with open('metrics_augment.json', 'w') as f:
      json.dump(metrics,f,indent=4)

evaluate('visual7w_data/compare_answers_augment.csv')