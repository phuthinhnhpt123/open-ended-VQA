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
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict

data_dir = "visual7w_data"

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

data = split_dataset(data_dir)

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

    return question_type_dict


a = stats_question_type(data,'train')
df = pd.DataFrame(a)
type_counts = df['types'].value_counts()
print(type_counts)

# df_why = df[df['types'] == 'why']
# print(df_why.head(10))

df_who = df[df['types'] == 'who']
df_why = df[df['types'] == 'why']
df_when = df[df['types'] == 'when']

sample_who = df_who.sample(n=100).reset_index(drop=True)
sample_why = df_why.sample(n=100).reset_index(drop=True)
sample_when = df_when.sample(n=100).reset_index(drop=True)

sample = pd.concat([sample_who,sample_why,sample_when])
print(sample)
