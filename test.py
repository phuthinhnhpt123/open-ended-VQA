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
    df_who = df[df['types'] == 'who']
    df_why = df[df['types'] == 'why']
    df_when = df[df['types'] == 'when']

    sample_who = df_who.sample(n=100)
    sample_why = df_why.sample(n=100)
    sample_when = df_when.sample(n=100)

    sample = pd.concat([sample_who,sample_why,sample_when]).reset_index(drop=True)

    sentences={"q_a":[]}
    for i in range(len(sample)):
        sentence=''
        sentence = sentence + 'q: ' + sample['questions'][i] + '. a: ' + sample['answers'][i]
        sentences['q_a'].append(sentence)
    d=pd.DataFrame(data=sentences)

    d.to_csv("sample.csv",index=False)
    print("Saved csv file")


data = split_dataset(data_dir)
# df = stats_question_type(data,'train')
# type_counts = df['types'].value_counts()
# print(type_counts)

# sample_dataset(df)


# train_data = read_pickle(data_dir,'train')
# test_data = read_pickle(data_dir,'test')
# val_data = read_pickle(data_dir,'val')

# print("num train data: ", len(train_data['questions']))
# print("num test data: ", len(test_data['questions']))
# print("num val data: ", len(val_data['questions']))

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

print("num train bf: ", count_questions(data['train']))
print("num test bf: ", count_questions(data['test']))
print("num val bf: ", count_questions(data['val']))

question_types_train = get_question_types(data['train'])
train_filtered = stratified_sample(data['train'], question_types_train)

question_types_test = get_question_types(data['test'])
test_filtered = stratified_sample(data['test'], question_types_test)

question_types_val = get_question_types(data['val'])
val_filtered = stratified_sample(data['val'], question_types_val)

train_data_dict = reformat_data(train_filtered)
test_data_dict = reformat_data(test_filtered)
val_data_dict = reformat_data(val_filtered)

# # Tạo các Dataset từ dữ liệu đã lọc
train_dataset = Dataset.from_dict(train_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)

print("num train af: ", count_questions(train_dataset))
print("num test af: ", count_questions(test_dataset))
print("num val af: ", count_questions(val_dataset))
# Tạo DatasetDict mới
# new_dataset = DatasetDict({
#     'train': train_dataset,
#     'test': test_dataset,
#     'val': val_dataset
# })

# print(new_dataset)




