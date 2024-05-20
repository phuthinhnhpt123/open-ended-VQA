from tqdm import tqdm
import torch
import json
import pandas as pd
from utils import generate_beam
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from evaluate import load
from torch.cuda.amp import autocast

    
def eval_vqa_open_ended(model, dataset):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    bert_score = load("bertscore")

    bleu_avg=0.
    bert_avg = 0.

    acc = 0.

    generated_answers = []
    bleu_scores = []
    bert_scores = []

    metrics = {"bleu":0.0, "Bertscore":0.0, "accuracy":0.0}

    for item in range(3):
        prefix, tokens, mask, q_len = dataset[item]

        prefix = prefix.to(device, dtype=torch.float32)
        tokens = tokens.to(device,dtype=torch.long)
        mask = mask.to(device,dtype=torch.long)

        with autocast(dtype=torch.float16):
          with torch.no_grad():
              embed = model.generate(prefix,tokens,mask,q_len).view(1,tokens.size(0),-1)

              out_text = model.gen_answer(prefix,embed,q_len)
              generated_answers.append(out_text)
              print(f'item {item}: ', out_text)

        # if out_text.lower()==dataset.answers[item].lower(): 
        #   acc+=1
            
        # reference = str(dataset.answers[item])
        # candidate = out_text

        # chencherry = SmoothingFunction()
        # bleu_1 = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))
        # bleu_scores.append(bleu_1)
        # bleu_avg+=bleu_1

        # a = bert_score.compute(references =[reference],predictions =[candidate],model_type = 'microsoft/deberta-large-mnli')
        # bert_scores.append(a['f1'][0])
        # bert_avg+= a['f1'][0]

    # print('------------')
    # print("BLEU {}".format(round(bleu_avg/len(dataset),3)))
    # print("BERTScore {}".format(round(bert_avg/len(dataset),3)))
    # print("Accuracy {}".format(round(acc/len(dataset),3)))
  	
    # metrics['bleu'] = round(bleu_avg/len(dataset),3)
    # metrics['Bertscore'] = round(bert_avg/len(dataset),3)
    # metrics['accuracy'] = round(acc/len(dataset),3)

    # compare_answer = {"predict": generated_answers,"answers": dataset.answers, "bleu_scores": bleu_scores, "bert_scores": bert_scores}