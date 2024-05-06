from train import pytorch_model_run
import torch
from predict import eval_gpt_open_ended
from models import VQAModel
from dataset import VqaDataset
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import pandas as pd
import pickle
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from evaluate import load

data_dir = "visual7w_data"

def count_questions(dataset):
    count = 0
    for item in dataset:
        count = count + len(item['qa_pairs'])
    return count

def split_dataset(data_dir):
    data = load_dataset("json", data_files=os.path.join(data_dir,"annotations", "dataset_v7w_telling.json"), field="images", split="train")

    train_dataset = data.filter(lambda x: x['split'] == 'train')
    test_dataset = data.filter(lambda x: x['split'] == 'test')
    val_dataset = data.filter(lambda x: x['split'] == 'val')

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'val': val_dataset
    })

    return dataset

def read_pickle(data_dir,split):
    data_path = data_dir +'/' +split +'.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def sample_dataset(df):
    df_what = df[df['types'] == 'what']
    df_why = df[df['types'] == 'why']
    # df_when = df[df['types'] == 'when']

    sample_what = df_what.sample(n=100)
    sample_why = df_why.sample(n=100)
    # sample_when = df_when.sample(n=100)

    sample = pd.concat([sample_what,sample_why]).reset_index(drop=True)

    sentences={"q_a":[]}
    for i in range(len(sample)):
        sentence=''
        sentence = sentence + 'q: ' + sample['questions'][i] + '. a: ' + sample['answers'][i]
        sentences['q_a'].append(sentence)
    d=pd.DataFrame(data=sentences)

    d.to_csv("sample.csv",index=False)
    print("Saved csv file")

def get_question_types(dataset):
    question_types = {}
    for item in dataset:
        for qa in item['qa_pairs']:
            q_type = qa['type']
            qa_id = qa['qa_id']
            if q_type not in question_types:
                question_types[q_type] = set()
            question_types[q_type].add(qa_id)
    return {key: list(values) for key, values in question_types.items()}

def stratified_sample(dataset, question_types, ratio=1/6):
    subset_ids = set()
    for q_type, ids in question_types.items():
        num_samples = int(len(ids) * ratio)
        chosen_ids = np.random.choice(ids, num_samples, replace=False)
        subset_ids.update(chosen_ids)

    new_items = []
    for item in dataset:
        filtered_qa_pairs = [qa for qa in item['qa_pairs'] if qa['qa_id'] in subset_ids]
        if filtered_qa_pairs:
            new_item = item.copy()
            new_item['qa_pairs'] = filtered_qa_pairs
            new_items.append(new_item)
    
    return new_items

def reformat_data(filtered_data):
    # Khởi tạo một dictionary với keys là các fields và values là empty lists
    reformatted_data = {key: [] for key in filtered_data[0].keys()}
    for item in filtered_data:
        for key in item:
            reformatted_data[key].append(item[key])
    return reformatted_data

# import pickle 

# dataset_path = "visual7w_data"
# with open(dataset_path + '/train.pkl', 'rb') as f:
#     train_data = pickle.load(f)
# with open(dataset_path + '/test.pkl', 'rb') as f:
#     test_data = pickle.load(f)
# with open(dataset_path + '/val.pkl', 'rb') as f:
#     val_data = pickle.load(f)

# train_df = pd.read_csv('visual7w_data/train.csv')
# test_df = pd.read_csv('visual7w_data/test.csv')
# val_df = pd.read_csv('visual7w_data/val.csv')

# grouped_train = train_df.groupby('types').size().reset_index(name='count')
# grouped_test = test_df.groupby('types').size().reset_index(name='count')
# grouped_val = val_df.groupby('types').size().reset_index(name='count')

# print(grouped_train)
# print(grouped_test)
# print(grouped_val)

df = pd.read_csv('visual7w_data/compare_answers.csv')
bleu_avg1=0.
bert_avg=0.
bert_score = load("bertscore")

for i in range(len(df['answers'])):
    reference = df['answers'][i]
    candidate = df['predict'][i]

    bert_avg+=bert_score.compute(references=[reference],predictions=[candidate],model_type = 'bert-base-uncased')['f1'][0]
    
    # chencherry = SmoothingFunction()
    # bleu_avg1+=sentence_bleu([reference.split()], candidate.split(), weights=(1,0,0,0), smoothing_function=chencherry.method7)

# print(round(bleu_avg1/len(df['answers']),3))
print(round(bert_avg/len(df['answers']),3))





