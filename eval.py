
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

    for i in range(len(df['answers'])):
        if df['predict'][i].lower()==df['answers'][i].lower(): 
            acc+=1

        bleu_avg+=float(df['bleu_scores'][i])
        bert_avg+=float(df['bert_scores'][i])

    return round(bleu_avg/len(df['answers']),3), round(bert_avg/len(df['answers']),3), round(acc/len(df['answers']),3)

def evaluate_result(result_dir):
    df = pd.read_csv(result_dir)

    metrics = {}

    for type in ['what', 'where', 'when', 'why', 'who', 'how']:
        sub_df = df[df['types'] == type].reset_index(drop=True)

        bleu, bert, acc = metrics_eval(sub_df)

        metrics[type] = [bleu, bert, acc]

        print(f'{type}: - number of questions: {len(sub_df)} -', metrics[type])
    
    with open('metrics_each_types_augment.json', 'w') as f:
      json.dump(metrics,f,indent=4)

evaluate_result('visual7w_data/compare_answers_augment.csv')