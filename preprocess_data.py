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
    img_idxs = []
    img_paths = []
    all_questions = []
    all_answers = []
    img_dict = {}

    for i in tqdm(range(len(data[split]))):
        img_id = data[split][i]['image_id']
        qa_pairs = data[split][i]['qa_pairs']

        img_path = data[split][i]['filename']
        filename = os.path.join(data_dir, "images", img_path)
        with torch.no_grad():
            prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()

        for j in range(len(qa_pairs)):
            if img_id not in img_dict.keys():
                img_dict[img_id] = [[qa_pairs[j]['question']],[qa_pairs[j]['answer']],prefix_i,filename]
            else:
                img_dict[img_id][0].append(qa_pairs[j]['question'])
                img_dict[img_id][1].append(qa_pairs[j]['answer'])
    
    for idx, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][2])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_idxs.append(idx)
            img_paths.append(img_dict[imgs][3])
    
    image_dict = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_idxs, "questions": all_questions, "answers": all_answers, "img_paths": img_paths}

    with open(out_path, 'wb') as f:
        pickle.dump(image_dict,f)
    print('Done') 

