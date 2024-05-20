
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

def eval_bad_result(result_dir):
    df = pd.read_csv(result_dir)

    bad_result = df[(df['bert_scores'] < 0.5)]

    bad_result = bad_result[(bad_result['types'] == 'why') | (bad_result['types'] == 'when')].reset_index(drop=True)

    pairs = []
    for i in range(len(bad_result)):
        sentence=''
        sentence = sentence + 'q: ' + bad_result['questions'][i] + '. a: ' + bad_result['answers'][i]
        pairs.append(sentence)
    
    bad_result['q_a'] = pairs

    bad_result.to_csv('bad_results.csv',index=False) 

def filter_question(result_dir):
    df = pd.read_csv(result_dir)

    df = df[df['questions'].str.contains('how many', case=False, na=False)]

    df.to_csv('howmany_questions.csv',index=False)

def sample_augment(result_dir):
    df = pd.read_csv(result_dir)

    words = ['one','1','0','none','zero']
    regex_pattern = '|'.join(words)
    
    df = df[(~df['answers'].str.contains(regex_pattern, case=False, na=False))].reset_index(drop=True)

    df.to_csv('how_many_questions.csv',index=False)

# evaluate_result('visual7w_data/data_gpt2/compare_answers_augment.csv')
# eval_bad_result('visual7w_data/data_gpt2/compare_answers.csv')
sample_augment('howmany_questions.csv')
# filter_question('visual7w_data/data_gpt2/train.csv')