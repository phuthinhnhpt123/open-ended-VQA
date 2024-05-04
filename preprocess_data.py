import torch
import skimage.io as io
import skimage.transform as transform
import torchvision
import clip
from PIL import Image
import pickle
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

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
    for _, ids in question_types.items():
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

    question_types_train = get_question_types(dataset['train'])
    train_filtered = stratified_sample(dataset['train'], question_types_train)

    question_types_test = get_question_types(dataset['test'])
    test_filtered = stratified_sample(dataset['test'], question_types_test)

    question_types_val = get_question_types(dataset['val'])
    val_filtered = stratified_sample(dataset['val'], question_types_val)

    train_data_dict = reformat_data(train_filtered)
    test_data_dict = reformat_data(test_filtered)
    val_data_dict = reformat_data(val_filtered)

    # Tạo các Dataset từ dữ liệu đã lọc
    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)
    val_dataset = Dataset.from_dict(val_data_dict)

    # Tạo DatasetDict mới
    new_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset
    })

    return new_dataset

def preprocess_data(data_dir, data, split, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    all_img_prefixes = []
    img_idxs = []
    img_paths = []
    all_questions = []
    all_answers = []
    all_types = []
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
                img_dict[img_id] = [[qa_pairs[j]['question']],[qa_pairs[j]['answer']],prefix_i,filename,[qa_pairs[j]['type']]]
            else:
                img_dict[img_id][0].append(qa_pairs[j]['question'])
                img_dict[img_id][1].append(qa_pairs[j]['answer'])
                img_dict[img_id][4].append(qa_pairs[j]['type'])
    
    for idx, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][2])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q].lower())
            all_answers.append(img_dict[imgs][1][q].strip(".").lower())
            img_idxs.append(idx)
            img_paths.append(img_dict[imgs][3])
            all_types.append(img_dict[imgs][4])
    
    image_dict = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_idxs, "questions": all_questions, "answers": all_answers, "img_paths": img_paths, "types":all_types}

    with open(out_path, 'wb') as f:
        pickle.dump(image_dict,f)
    print('Done') 

def update_classes(pkl_train, pkl_val, pkl_test):
    # standardize answer ids across datasets and compute the maximum number of generated output tokens based on the train set
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    with open(pkl_train, 'rb') as f:
        data_train = pickle.load(f)
    with open(pkl_val, 'rb') as f:
        data_val = pickle.load(f)
    with open(pkl_test, 'rb') as f:
        data_test = pickle.load(f)
    
    cur_id = 0
    class_names_list = []
    class_ids_list = [[],[],[]]

    for i, data in enumerate([data_train,data_val,data_test]):
        for answer in data['answers']:
            if answer not in class_names_list:
                class_names_list.append(answer)
                class_ids_list[i].append(cur_id)
                cur_id+=1
            else:
                class_ids_list[i].append(class_names_list.index(answer))

    for _, data in enumerate([data_train,data_val,data_test]):

        q_lens = []
        a_lens = []

        for question in data['questions']:
            q_lens.append(len(tokenizer.encode(question)))
        for answer in data['answers']:
            a_lens.append(len(tokenizer.encode(str(answer))))
        data['max_seqs_len']=(int(np.mean(q_lens)+3*np.std(q_lens)),int(np.mean(a_lens)+3*np.std(a_lens)))
          
    data_train['class_ids'] = class_ids_list[0]
    data_val['class_ids'] = class_ids_list[1]
    data_test['class_ids'] = class_ids_list[2]
    
    with open(pkl_train, 'wb') as f:
        pickle.dump(data_train,f)
    with open(pkl_val, 'wb') as f:
        pickle.dump(data_val,f)
    with open(pkl_test, 'wb') as f:
        pickle.dump(data_test,f)

