import torch
# import skimage.io as io
# import skimage.transform as transform
# import torchvision
# import clip
import pandas as pd
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import string
import random
import numpy as np
# from transformers import set_seed, GPT2Config, GPT2Tokenizer
from datasets import load_dataset, Dataset, DatasetDict

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

    # print(dataset['train'][0])
    return dataset

def preprocess_data(data_dir, data, split, out_path):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    all_img_prefixes = []
    img_ids = []
    img_paths = []
    all_questions = []
    all_answers = []

    for i in tqdm(range(len(data[split]))):
        qa_pairs = data[split][i]['qa_pairs']

        img_path = data[split][i]['filename']
        filename = os.path.join(data_dir, "images", img_path)
        with torch.no_grad():
            prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()
        all_img_prefixes.append(prefix_i)

        for j in range(len(qa_pairs)):
            img_ids.append(qa_pairs[j]['image_id'])
            img_paths.append(img_path)
            all_questions.append(qa_pairs[j]['question'])
            all_answers.append(qa_pairs[j]['answer'])
    
    image_dict = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions, "answers": all_answers, "img_paths": img_paths}

    with open(out_path, 'wb') as f:
        pickle.dump(image_dict,f)
    print('Done')


# dataset = split_dataset()
# train_dict = preprocess_data(dataset,'train')
# print(train_dict['img_ids'][0:10])  
# print(train_dict['questions'][0:10])
# # split_dataset()   

def main():
    # dataset = split_dataset(data_dir)
    # for split in ['train','test','val']:
    #     out_path = data_dir + "/{}.pkl".format(split)
    #     preprocess_data(data_dir, dataset,split,out_path)
    with open("visual7w_data/train.pkl",'rb') as f:
        train_data = pickle.load(f)
    with open("visual7w_data/test.pkl",'rb') as f:
        test_data = pickle.load(f)
    with open("visual7w_data/val.pkl",'rb') as f:
        val_data = pickle.load(f)
    

    


# data_dir = '/kaggle/input/visual7w/visual7w_data'
main()

