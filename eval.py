
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
    
    with open('metrics_each_types.json', 'w') as f:
      json.dump(metrics,f,indent=4)

def eval_bad_result(result_dir):
    df = pd.read_csv(result_dir)

    for type in ['what', 'where', 'when', 'why', 'who', 'how']:
        sub_df = df[df['types'] == type].reset_index(drop=True)

        # bad_result = sub_df[sub_df['bert_scores'] < 0.5]

        print(f'{type}: - number of questions: {len(sub_df)}')

    # bad_result = df[(df['bert_scores'] < 0.5)]

    # bad_result = bad_result[(bad_result['types'] == 'what')].reset_index(drop=True)

    # bad_result.to_csv('bad_results_what.csv',index=False) 

def filter_question(result_dir,destination_dir):
    df = pd.read_csv(result_dir)
    des_df = pd.read_csv(destination_dir)

    check_question = list(des_df['questions'])
    check_answer = list(des_df['answers'])

    indices_to_drop = []
    for i in range(len(df)):
        if df['questions'][i] in check_question and df['answers'][i] in check_answer:
            idx = df[(df['questions'] == df['questions'][i]) & (df['answers'] == df['answers'][i])].index
            indices_to_drop.extend(idx)
    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv('what_question_samples.csv',index=False)

def sample_augment(result_dir):
    df = pd.read_csv(result_dir)

    # words = ['one','1','0','none','zero']
    # regex_pattern = '|'.join(words)
    
    # df = df[(~df['answers'].str.contains(regex_pattern, case=False, na=False))].reset_index(drop=True)
    df = df[df['types'] == 'what']

    sub_df = df.sample(n=238, random_state=1)

    print(len(sub_df['answers'].unique()))
    sub_df.to_csv('what_question_samples.csv',index=False)

evaluate_result('visual7w_data/data_gpt2/evaluation/compare_answers(3).csv')
# eval_bad_result('augment.csv')
# sample_augment('visual7w_data/data_gpt2/dataset/train/train.csv')
# filter_question('what_question_samples.csv','augment.csv')