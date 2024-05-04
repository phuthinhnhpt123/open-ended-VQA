from train import pytorch_model_run
import torch
from predict import eval_gpt_open_ended
from models import VQAModel
from dataset import VqaDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
import os
import pandas as pd
import pickle
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict

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

def stats_question_type(data, split):
    questions = []
    answers = []
    types = []
    sub_data = data[split]
    for i in range(len(sub_data)):
        qa_pairs = data[split][i]['qa_pairs']
        for j in range(len(qa_pairs)):
            questions.append(qa_pairs[j]['question'])
            answers.append(qa_pairs[j]['answer'])
            types.append(qa_pairs[j]['type'])
    
    question_type_dict = {"questions": questions, "answers": answers, "types": types}

    return pd.DataFrame(question_type_dict)

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


# data = split_dataset(data_dir)
# df = stats_question_type(data,'train')
# sample_dataset(df)

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

# dataset_path = 'visual7w_data'

# train_dataset = VqaDataset(dataset_path+'/',split="train",model_type='gpt2')
# test_dataset = VqaDataset(dataset_path+'/',split="test",model_type='gpt2')

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)

# for item in range(10):
#     prefix, labels, tokens, mask, q_len = train_dataset[item]

#     print(f"tokens size: {tokens.size()}")
#     print(f"\nmask: {mask}")
#     print(f"\nq_len: {q_len}")

# dataset = {'questions': test_dataset.questions, 'answers': test_dataset.answers}
# df = pd.DataFrame(dataset)
# print(df.head(10))

import pickle 

dataset_path = "visual7w_data"
with open(dataset_path + '/train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(dataset_path + '/test.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open(dataset_path + '/val.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(train_data.keys())

df = pd.DataFrame({
    'img_ids': train_data['img_ids'],
    'questions': train_data['questions'],
    'answers': train_data['answers'],
    'img_paths': train_data['img_paths'],
    'class_ids': train_data['class_ids']
})
df.to_csv('train.csv',index=False)

