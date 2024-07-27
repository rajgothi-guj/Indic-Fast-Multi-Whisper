# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29400 finetuning/run_finetuning.py 

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import sys
current_directory = os.getcwd()
sys.path.append(current_directory)

import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from dataloader import get_dataloader

import yaml
import argparse
import wandb

import os
local_rank = int(os.environ["LOCAL_RANK"])

def load_config(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

def print_chosen_variables(args):
    for key, value in vars(args).items():
        print(f"{key}: {value}")

def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    train_only_decoder: bool,
    max_grad_norm: float,
) -> Tuple[float, Iterator]:
    model.train()
    total_loss = 0
    for _ in range(accum_grad_steps):
        x, y_in, y_out = next(train_iter)
        # x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        # x, y_in, y_out = x.cuda(), y_in.cuda(), y_out.cuda()

        if train_only_decoder:
            with torch.no_grad():
                audio_features = model.embed_audio(x)
        else:
            audio_features = model.embed_audio(x)
        logits = model.logits(y_in, audio_features=audio_features)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        loss = loss / accum_grad_steps
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss


@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader) -> float:
    model.eval()
    total_loss = 0
    for x, y_in, y_out in tqdm(dev_loader):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        # x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)

        # print(tokenizer.decode(y_in[0]))

        logits = model(x, y_in)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)
        total_loss += loss.item()
    return total_loss / len(dev_loader)


# def save_model(model: Whisper, save_path: str) -> None:
#     # save model in half precision to save space
#     model = copy.deepcopy(model).half()
#     # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
#     torch.save({"model_state_dict": model.state_dict(), "dims": asdict(model.dims)}, save_path)

def save_checkpoint(model: Whisper, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LambdaLR, step: int, save_path: str, args: argparse.Namespace) -> None:
    # Save model, optimizer, and scheduler states along with the current step
    if args.use_multi_gpu:
        torch.save({
            'step': step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            "dims": asdict(model.dims)
        }, save_path)

    else:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            "dims": asdict(model.dims)
        }, save_path)


def load_checkpoint(model: Whisper, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LambdaLR, load_path: str) -> int:
    # Load model, optimizer, and scheduler states along with the current step
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'] + 1  # Return the next step to resume training


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_args(args, save_path):
    with open(save_path, "w") as f:
        yaml.safe_dump(vars(args), f)


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch


def main_loop(
    model: Whisper,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
) -> None:
    if args.start_from_checkpoint:
        #load model        
        start_step = load_checkpoint(model,optimizer,scheduler,args.chekpoint_path)

    min_loss = 1e9
    # min_loss = evaluate(model, dev_loader)
    # print(f"Initial loss: {min_loss}")
    # wandb.log({'Validation Loss':min_loss})

    start_step = 1

    pbar = tqdm(range(start_step, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.train_only_decoder,
            args.max_grad_norm,
        )
        torch.cuda.empty_cache()
        pbar.set_postfix({"loss": train_loss})

        if step % args.log_steps == 0:
            wandb.log({'Train Loss':train_loss})

        if step % args.eval_steps == 0:
            eval_loss = evaluate(model, dev_loader)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            wandb.log({'Validation Loss':eval_loss})

            if eval_loss < min_loss:
                min_loss = eval_loss
                save_checkpoint(model, optimizer=optimizer,scheduler=scheduler,step=step, save_path=f"{args.save_dir}/best_model.pt",args=args)

            if args.save_all_checkpoints:
                save_checkpoint(model, optimizer=optimizer,scheduler=scheduler,step=step,save_path=f"{args.save_dir}/step{step}.pt",args=args)

            save_checkpoint(model,optimizer=optimizer,scheduler=scheduler,step=step, save_path=f"{args.save_dir}/last_model.pt",args=args)

class MyDataParallel(nn.DataParallel):
# class MyDataParallel(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def main():

    args = load_config("Config/train/multilingual.yaml")
    
    wandb.init(
        entity = 'rajsony',
        project = 'multilingual',
        name = f'{args.language}_{args.save_dir}',
        config = args
    )

    print_chosen_variables(args)
    # args = get_parser().parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    save_args(args, f"{args.save_dir}/args.yaml")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # add new tokens and resize the embeddings...
    if args.add_new_tokens:
        # Read JSON file
        with open(args.add_new_token_json_path, 'r',encoding='utf-8') as json_file:
            decoded_tokens = json.load(json_file)

        tokenizer = get_tokenizer(multilingual=True,add_new_tokens=tuple(decoded_tokens))
        model = whisper.load_model(args.model, device)
        model.resize_token_embeddings(tokenizer.total_token)
        print("Total Token: ",tokenizer.total_token)

        file_path = f"{args.save_dir}/add_tokens.json"

        # Write list to JSON file
        with open(file_path, 'w') as json_file:
            json.dump(decoded_tokens, json_file,ensure_ascii=False)

    else:
        tokenizer = get_tokenizer(multilingual=True, task="transcribe")
        model = whisper.load_model(args.model, device)
    
    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    if args.use_multi_gpu:
        # model = nn.DataParallel(model)
        model = MyDataParallel(model,device_ids=[0,1])
        # print(model.module)
        model.to(device)

    # print(device.)
    fp16 = True
    train_loader = get_dataloader(
        json=args.train_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=args.prompt_use_rate,
        no_timestamps_rate=args.no_timestamps_rate,
        shuffle=True,
    )
    dev_loader = get_dataloader(
        json=args.dev_json,
        tokenizer=tokenizer,
        batch_size=args.dev_batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        # always use prompts and timestamps for validation to make it deterministic
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
        shuffle=False,
    )
    if args.use_adam_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("For using Adam 8bit optimizer you need to have bitsandbytes installed.")
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    args.train_steps = args.epoch * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    main_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
    )

    wandb.finish()

if __name__ == "__main__":
    main()
