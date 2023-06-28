import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from model import AutoCrossEncoder
from batch_preprocessing import batchify_dataset

import torch
import os
import shutil
import argparse

def save_checkpoint(model, save_dir, total_steps, device):
    """
    :param model: (obj) model
    :param save_dir: (str) checkpoint save directory
    :param epoch: (int) checkpoint epoch
    :param device: specify device
    :return: None
    """

    print(f"saving model checkpoint as ckpt_{total_steps}.pt ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.cpu()  # TODO: this line doesn't seem to work when training with TPU
    torch.save(model.state_dict(), os.path.join(save_dir, f"ckpt_{total_steps}.pt"))
    model.to(device)
    print("checkpoint saved!")

def train(
    model, 
    dataloader_train,
    dataloader_eval,
    device,
    epochs,
    learning_rate,
    weight_decay,
    num_warmup_steps,
    accumulation_steps,
    best_model_dir,
    eval_every,
    patience
):
    
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    
    optimizer = AdamW(params=model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay, correct_bias=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader_train)*epochs,
    )
    
    total_steps = 0
    patience_counter = 0
    best_eval_loss = np.inf

    for epoch in range(epochs):
        print(f"epoch: {epoch}")

        model.train()

        total_epoch_loss = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)            

        for i, batch in enumerate(progress_bar):
            total_steps += 1

            if total_steps % eval_every == 0:
            # run model on the evaluation set

                model.eval()
                eval_progress_bar = tqdm(dataloader_eval, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
                batched_eval_losses = []
                with torch.no_grad():

                    for j, eval_batch in enumerate(eval_progress_bar):
                        input_ids = eval_batch["input_ids"].to(device)
                        attention_mask = eval_batch["attention_mask"].to(device)
                        token_type_ids = eval_batch["token_type_ids"].to(device)
                        labels = eval_batch["labels"].to(device)

                        curr_eval_loss = model.forward(
                            input_ids,
                            attention_mask,
                            token_type_ids,
                            labels=labels
                        )
                        batched_eval_losses.append(curr_eval_loss.cpu().numpy())
    
                    mean_eval_loss = np.mean(batched_eval_losses)
                    print(f"eval loss: {mean_eval_loss} after {total_steps} training steps")

                    if mean_eval_loss < best_eval_loss:
                        best_eval_loss = mean_eval_loss
                        patience_counter = 0
                        
                        print("curr eval loss is better, saving updated current best")
                        model.model.save_pretrained(os.path.join(best_model_dir, f"crossencoder"))
                        
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Max patience of {patience} exceeded, breaking training")
                            return
                        
                # do stuff here #
                model.train()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            loss = model.forward(
                input_ids,
                attention_mask,
                token_type_ids,
                labels=labels
            )
            
            loss = loss / accumulation_steps            
            total_epoch_loss += loss.item()

            loss.backward()

            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(dataloader_train)):
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({'training_loss': '{:.4f}'.format(loss.item())})

        tqdm.write(f"\nEpoch {epoch}")
        average_loss = total_epoch_loss / len(dataloader_train)
        tqdm.write(f"Training loss: {average_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing arguments')
    
    parser.add_argument('--model_name_or_path', metavar='N', type=str, default='bert-base-uncased',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--train_path', metavar='N', type=str, default=None,
                        help='copy with cite dataset (csv file with query, positive, negative columns)')
    parser.add_argument('--best_model_dir', metavar='N', type=str, default='./best_model_dir',
                        help='Directory location to save out the best model')
    parser.add_argument('--num_epochs', metavar='N', type=int, default=10,
                        help='Max number of training steps (Number of global batches to run)')
    parser.add_argument('--batch_size', metavar='N', type=int, default=16,
                        help='Per device batch size') 
    parser.add_argument('--eval_batch_size', metavar='N', type=int, default=16,
                        help='Per device eval batch size')        
    parser.add_argument('--gradient_accumulation_steps', metavar='N', type=int, default=4,
                        help='Gradient accumulation steps')        
    parser.add_argument('--learning_rate', metavar='N', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay_rate', metavar='N', type=float, default=0.00,
                        help='Learning rate')
    parser.add_argument('--num_warmup_steps', metavar='N', type=int, default=500,
                        help='Number of warmup steps in scheduler')
    parser.add_argument('--eval_every', metavar='N', type=int, default=500,
                        help='Number of warmup steps in scheduler')
    parser.add_argument('--patience', metavar='N', type=int, default=5,
                        help='Patience parameter of when to break training')  
    
    # DEVICE/GPUS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    args, unknown = parser.parse_known_args()

    # MODEL/TOKENIZER  
    model = AutoCrossEncoder(args.model_name_or_path)
    print("Number of trainable parameters ...")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("putting model on the gpu ...")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # DATA/DATALOADER

    print(f"reading csv from {args.train_path}")
    df = pd.read_csv(args.train_path)
    train_df, eval_df = train_test_split(df, test_size=0.1 ,shuffle=True, random_state=3)

    dataloader_train = batchify_dataset(
        train_df,
        tokenizer,
        max_length=256,
        batch_size=args.batch_size
    )

    dataloader_eval = batchify_dataset(
        eval_df,
        tokenizer,
        max_length=256,
        batch_size=args.eval_batch_size
    )

    print("length of dataloaders")
    print(f"train: {len(dataloader_train)}")
    print(f"eval: {len(dataloader_eval)}")

    train(
        model,
        dataloader_train,
        dataloader_eval,
        device,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay_rate,
        num_warmup_steps=args.num_warmup_steps,
        accumulation_steps=args.gradient_accumulation_steps,
        best_model_dir=args.best_model_dir,
        eval_every=args.eval_every,
        patience=args.patience,
    )
