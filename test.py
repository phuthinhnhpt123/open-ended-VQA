from train import pytorch_model_run
import torch
from predict import eval_gpt_open_ended
# import clip
from PIL import Image
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

def preprocess_augment_data(data_dir, augment_data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    with open(data_dir+'/train.pkl','rb') as f:
        data_train = pickle.load(f)
    
    df_augment = pd.read_csv(augment_data_dir)

    img_dict = {}
    all_img_prefixes=data_train['img_prefix']
    all_questions = data_train['questions']
    all_answers = data_train['answers']
    img_idxs = data_train['img_ids']
    img_paths = data_train['img_paths']
    all_types = data_train['types']
    img_prefixes = []

    for i in range(len(df_augment['questions'])):
        img_id = df_augment['image_id'][i]
        img_path = os.path.join(data_dir,'images_use',img_id)
        question = str(df_augment['questions'][i]).lower()
        answer = str(df_augment['answers'][i]).strip(".").lower()
        question_type = df_augment['types'][i]

        if img_id not in img_dict.keys():
            with torch.no_grad():
                prefix_i = clip_model.encode_image(preprocess(Image.open(img_path)).unsqueeze(0).to(device)).cpu()
            img_dict[img_id] = [[question],[answer],prefix_i,img_id,[question_type]]
        else:
            img_dict[img_id][0].append(question)
            img_dict[img_id][1].append(answer)
            img_dict[img_id][3].append(question_type)

    last_idx = int(data_train['img_ids'][-1]) + 1
    for idx, imgs in enumerate(img_dict.keys()):
        img_prefixes.append(img_dict[imgs][2])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_idxs.append(idx+last_idx)
            img_paths.append(img_dict[imgs][3])
            all_types.append(img_dict[imgs][4][q])
    
    addition_img_prefixes = torch.tensor(img_prefixes)
    image_dict = {"img_prefix": torch.cat((all_img_prefixes,addition_img_prefixes), dim=0), "img_ids": img_idxs, "questions": all_questions, "answers": all_answers, "img_paths": img_paths, "types":all_types}

    df = pd.DataFrame.from_dict(data=image_dict)
    df.to_csv('test.csv',index=False)

preprocess_augment_data('visual7w_data','visual7w_data/augment.csv')


