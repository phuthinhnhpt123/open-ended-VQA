from train import pytorch_model_run
import torch
from predict import eval_gpt_open_ended
from models import VQAModel
from dataset import VqaDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
from torch.cuda.amp import autocast
from utils import generate_beam
import os

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2-xl")
    parser.add_argument("--setting", type=str, default="lora", choices=("lora", "frozen"))
    parser.add_argument("--mapping_type", type=str, default="MLP")
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument(
        "--dataset_path", type=str, default="../visual7w/"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--validation_step", type=int, default=1000)
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--eval", dest="eval", action="store_true")

    parser.add_argument("--verbose", dest="verbose", action="store_true")

    args = parser.parse_args()
    
    set_random_seeds(args.seed)
    return args

def print_nearest_text_token(vis_token, model):
    """print the nearest token in the vocabulary to the given token through model.gpt.embeddings.weight"""
    embeddings = model.gpt.transformer.wte.weight
    distances = torch.norm(embeddings - vis_token, dim=1)
    nearest_token_idx = torch.argmin(distances)
    print(model.tokenizer.decode([nearest_token_idx.item()])) 

if __name__ == "__main__":
    args = parse_argument()
    suffix = f"v5_prefixlength_{args.prefix_length}_seed_{args.seed}_gpttype_{args.model_type.replace('/','')}_setting_{args.setting}"

    args.out_dir = os.path.join('../checkpoints', suffix)
    train_dataset = VqaDataset(args.dataset_path+'/',split="train",prefix_length=args.prefix_length,model_type=args.model_type)
    val_dataset = VqaDataset(args.dataset_path+'/',split="val",prefix_length=args.prefix_length,model_type=args.model_type)
    test_dataset = VqaDataset(args.dataset_path+'/',split="test",prefix_length=args.prefix_length,model_type=args.model_type,like_test=True)

    model = VQAModel(
        prefix_length=args.prefix_length,
        clip_length=4,
        setting=args.setting,
        mapping_type=args.mapping_type,
        args=args,
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)


    if not args.eval:
        model = pytorch_model_run(train_dataloader, val_dataloader, model, args)
    else:
        checkpoint = os.path.join(args.out_dir, f"open_ended_latest.pt")
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!")
        if os.path.exists(checkpoint):
            model.load_state_dict(
                torch.load(checkpoint, map_location=torch.device("cpu")), strict=False
            )
        else:
            raise ValueError("Please provide valid path for loading checkpoint")
        
        print_vis_token_meaning = True
        for item in range(0,10):
            prefix,  labels, tokens, mask, q_len = test_dataset[item]
            prefix = prefix.type(torch.float32)
            tokens = tokens.type(torch.long)
            mask = mask.type(torch.long)
            with autocast(dtype=torch.float16):
                with torch.no_grad():
                    embed = model.generate(prefix,labels,tokens,mask,q_len).view(1,tokens.size(0),-1)
                    if print_vis_token_meaning:
                        prefix_projections = embed[:,q_len:q_len+model.prefix_length,:]
                        for i in range(prefix_projections.size(1)):
                          print_nearest_text_token(prefix_projections[0,i], model)
                    out_text = generate_beam(model, model.tokenizer,generated=embed,entry_length=test_dataset.max_seqs_len[1], temperature=1)[0]
                    print(out_text)
                    print(test_dataset.answers[item])

   