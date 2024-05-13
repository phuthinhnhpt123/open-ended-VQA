from tqdm import tqdm
import torch
import json
import pandas as pd
from utils import generate_beam
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from eval import load
from torch.cuda.amp import autocast

    
def eval_gpt_open_ended(model, dataset, args):
    model.eval()

    bert_score = load("bertscore")

    bleu_avg=0.
    bert_avg = 0.

    acc = 0.

    generated_answers = []

    metrics = {"bleu":0.0, "Bertscore":0.0, "accuracy":0.0}

    with tqdm(total=len(dataset)) as epoch_pbar:
        epoch_pbar.set_description("Testing")
        for item in range(len(dataset)):
            prefix, tokens, mask, q_len = dataset[item]

            prefix = prefix.type(torch.float32)
            tokens = tokens.type(torch.long)
            mask = mask.type(torch.long)

            with autocast(dtype=torch.float16):
              with torch.no_grad():
                  embed = model.generate(prefix,tokens,mask,q_len).view(1,tokens.size(0),-1)

                  out_text = generate_beam(model, model.tokenizer,generated=embed,entry_length=dataset.max_seqs_len[1], temperature=1)[0]
                  generated_answers.append(out_text)
                  print(f'item {item}: ', out_text)

            if out_text.lower()==dataset.answers[item].lower(): 
              acc+=1
                
            reference = str(dataset.answers[item])
            candidate = out_text

            # chencherry = SmoothingFunction()
            bleu_1 = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))
            bleu_avg+=bleu_1

            a = bert_score.compute(references =[reference],predictions =[candidate],model_type = 'bert-base-uncased')
            bert_avg+= a['f1'][0]

    print('------------')
    print("BLEU {}".format(round(bleu_avg/len(dataset),3)))
    print("BERTScore {}".format(round(bert_avg/len(dataset),3)))
    print("Accuracy {}".format(round(acc/len(dataset),3)))
  	
    metrics['bleu'] = round(bleu_avg/len(dataset),3)
    metrics['Bertscore'] = round(bert_avg/len(dataset),3)
    metrics['accuracy'] = round(acc/len(dataset),3)

    with open('metrics.json', 'w') as f:
      json.dump(metrics,f,indent=4)

    compare_answer = {"predict": generated_answers,"answers": dataset.answers}
    df = pd.DataFrame(data=compare_answer)
    df.to_csv('compare_answers.csv',index=False)

    # return generated_answers