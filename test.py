from train import pytorch_model_run
import torch
# import clip
from PIL import Image
from models import VQAModel
from dataset import VqaDataset
from torch.utils.data import DataLoader
import numpy as np
import json
import random
import os
import pandas as pd
import pickle
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

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
    # df_what = df[df['types'] == 'what']
    df_why = df[df['types'] == 'why']
    # df_when = df[df['types'] == 'when']
    df_how = df[df['types'] == 'how']

    # sample_what = df_what.sample(n=100)
    sample_why = df_why.sample(n=100)
    # sample_when = df_when.sample(n=100)
    sample_how = df_how.sample(n=100)

    sample = pd.concat([sample_how,sample_why]).reset_index(drop=True)

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


def pad_sequences(tokenizer, question,answer):
    max_seqs_len = [12,6]
    prefix_len = 8

    m = [torch.tensor(tokenizer.encode('question: ')),torch.tensor(tokenizer.encode('context: ')),torch.tensor(tokenizer.encode('answer: ')),torch.tensor(tokenizer.encode('<|endoftext|>'))]
    m_mask = [torch.ones(len(tokenizer.encode('question: '))),torch.ones(len(tokenizer.encode('context: '))),torch.ones(len(tokenizer.encode('answer: '))),torch.zeros(len(tokenizer.encode('<|endoftext|>')))]
    q=torch.tensor(tokenizer.encode(question))
    a=torch.tensor(tokenizer.encode(str(answer)))
    
    q,q_mask,leftover_tokens = make_padding(max_seqs_len[0],q,question=True)
    print('q: ', q)
    q_len = m[0].size(0) + q.size(0) + m[1].size(0)
    a,a_mask,_ = make_padding(max_seqs_len[1],a,leftover_tokens=leftover_tokens)
    print('a: ', a)
    if len((a==0).nonzero())!=0:
        pad_start = (a==0).nonzero()[0]
    else:
        pad_start=[]
    a = torch.cat((a,m[3])) if len(pad_start)==0 else torch.cat((a[:pad_start],m[3],a[pad_start:]))
    sentence_tokens = torch.cat((m[0],q,m[1],torch.ones(prefix_len),m[2],a))
    
    sentence_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(prefix_len),m_mask[2],a_mask,m_mask[3]))
    
    return sentence_tokens,sentence_mask, q_len

def make_padding(max_len, tokens, question=False,leftover_tokens=0):
    padding = max_len - tokens.size(0) 
    if padding > 0:
        if question:
            leftover_tokens = padding
            mask = torch.ones(tokens.size(0))
        else:
            tokens = torch.cat((tokens, torch.zeros(padding+leftover_tokens)))
            mask = torch.zeros(max_len+leftover_tokens)    
            
    elif padding==0:
        if question:
            mask = torch.ones(tokens.size(0)) 
        else:
            mask = torch.zeros(tokens.size(0)+leftover_tokens)
            tokens = torch.cat((tokens,torch.zeros(leftover_tokens)))
                
    elif padding < 0:
        if question:
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
        else:
            tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
            mask = torch.zeros(max_len+ leftover_tokens)
    return tokens, mask, leftover_tokens

# import shutil

# with open('visual7w_data/train.pkl','rb') as f:
#     data_train = pickle.load(f)
# with open('visual7w_data/test.pkl','rb') as f:
#     data_test = pickle.load(f)
# with open('visual7w_data/val.pkl','rb') as f:
#     data_val = pickle.load(f)

# img_paths = data_train['img_paths'] + data_test['img_paths'] + data_val['img_paths']
# img_paths_unique = set(img_paths)

# src_img = 'visual7w_data/images'
# destination_img = 'visual7w_data/images_use'

# for img in img_paths_unique:
#     path = os.path.join(src_img,img)

#     shutil.copy(path,destination_img)

# tokenizer = AutoTokenizer.from_pretrained('gpt2')

# for i in range(3):
#     question = data['questions'][i]
#     answer = data['answers'][i]

#     print('question: ', question)
#     print('answer: ', answer)

#     sentence_tokens,sentence_mask, q_len = pad_sequences(tokenizer,question,answer)

#     print('sentences tokens: ', sentence_tokens)
#     print('q len: ', q_len)

