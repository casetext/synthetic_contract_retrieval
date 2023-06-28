import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from model import AutoDualEncoder
from batch_preprocessing import batchify_dataset

import torch
import os
import shutil

import argparse
import ipdb

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

## MAIN TRAINING LOOP
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
    eval_every,
    patience,
    best_model_dir,
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

    # best_micro_average_precision = 0.0

    for epoch in range(epochs):
        print(f"epoch: {epoch}")

        model.train()

        total_epoch_loss = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)            

        for i, batch in enumerate(progress_bar):
            total_steps += 1

            ## EVALUATION CODE BASED ON EVAL LOSS
            if total_steps % eval_every == 0:
            # run model on the evaluation set

                model.eval()
                eval_progress_bar = tqdm(dataloader_eval, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
                batched_eval_losses = []
                with torch.no_grad():

                    for j, eval_batch in enumerate(eval_progress_bar):
                        anchor_input_ids = eval_batch["anchor_input_ids"].to(device)
                        anchor_attention_mask = eval_batch["anchor_attention_mask"].to(device)
                        positive_input_ids = eval_batch["positive_input_ids"].to(device)
                        positive_attention_mask = eval_batch["positive_attention_mask"].to(device)
                        negative_input_ids = eval_batch["negative_input_ids"].to(device)
                        negative_attention_mask = eval_batch["negative_attention_mask"].to(device)

                        curr_eval_loss = model.forward_triplet(
                            anchor_input_ids,
                            anchor_attention_mask,
                            positive_input_ids,
                            positive_attention_mask,
                            negative_input_ids,
                            negative_attention_mask,
                            margin=0.1
                        )
                        
                        batched_eval_losses.append(curr_eval_loss.item())
                        
                    mean_eval_loss = np.mean(batched_eval_losses)
                    print(f"eval loss: {mean_eval_loss} after {total_steps} training steps")

                    if mean_eval_loss < best_eval_loss:
                        best_eval_loss = mean_eval_loss
                        patience_counter = 0

                        print("curr eval loss is better, saving updated current best")
                        # overwrite the previous best and replace it with the current best
                        print("overwriting current question encoder ...")
                        print("overwriting current passage encoder ...")
                        print("deleted previous best model, saving new best")

                        model.model.save_pretrained(os.path.join(best_model_dir, f"question_encoder"))
                        model.passage_encoder.save_pretrained(os.path.join(best_model_dir, f"passage_encoder"))
                    
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Max patience of {patience} exceeded, breaking training")
                            return

                model.train()

            anchor_input_ids = batch["anchor_input_ids"].to(device)
            anchor_attention_mask = batch["anchor_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            negative_input_ids = batch["negative_input_ids"].to(device)
            negative_attention_mask = batch["negative_attention_mask"].to(device)

            loss = model.forward_triplet(
                anchor_input_ids,
                anchor_attention_mask,
                positive_input_ids,
                positive_attention_mask,
                negative_input_ids,
                negative_attention_mask,
                margin=0.1
            )
            
            loss = loss / accumulation_steps            
            total_epoch_loss += loss.item()

            loss.backward()

            # gradient descent step for {optimizer, scheduler}
            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(dataloader_train)):
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({'training_loss': '{:.4f}'.format(loss.item())})

        tqdm.write(f"\nEpoch {epoch}")
        average_loss = total_epoch_loss / len(dataloader_train)
        tqdm.write(f"Training loss: {average_loss}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing arguments')
    
    parser.add_argument('--model_name_or_path', metavar='N', type=str, default='nlpaueb/legal-bert-base-uncased',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--freeze', action=argparse.BooleanOptionalAction,
                        help='Whether or not to freeze the passage encoder')
    parser.add_argument('--train_path', metavar='N', type=str, default=None,
                        help='copy with cite dataset (csv file with query, positive, negative columns)')
    parser.add_argument('--best_model_dir', metavar='N', type=str, default='./best_model_dir',
                        help='Directory location to save out the best model')            
    parser.add_argument('--num_epochs', metavar='N', type=int, default=20,
                        help='Max number of training steps (Number of global batches to run)')
    parser.add_argument('--batch_size', metavar='N', type=int, default=16,
                        help='Per device batch size') 
    parser.add_argument('--eval_batch_size', metavar='N', type=int, default=16,
                        help='Per device eval batch size')        
    parser.add_argument('--gradient_accumulation_steps', metavar='N', type=int, default=4,
                        help='Gradient accumulaiton steps')        
    parser.add_argument('--learning_rate', metavar='N', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay_rate', metavar='N', type=float, default=0.00,
                        help='Weight decay rate')
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
    model = AutoDualEncoder.from_pretrained(args.model_name_or_path)

    print("num trainable parameters ...")

    if not args.freeze:
        model.load_passage_encoder(args.model_name_or_path)
    else:
        model.load_passage_encoder(args.model_name_or_path, freeze=True)

    print("Number of trainable parameters ...")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("putting model on the gpu ...")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # DATA/DATALOADER

    print(f"reading csv from {args.train_path}")
    df = pd.read_csv(args.train_path)
    train_df, eval_df = train_test_split(df, test_size=0.05, shuffle=True, random_state=3)

    print(len(train_df))
    print(len(eval_df))

    dataloader_train = batchify_dataset(
        train_df,
        tokenizer,
        max_length=128,
        batch_size=args.batch_size
    )

    dataloader_eval = batchify_dataset(
        eval_df,
        tokenizer,
        max_length=128,
        batch_size=args.eval_batch_size,
        sequential_sampler=True
    )
    # ipdb.set_trace()

    print("length of dataloader")
    print(f"train: {len(dataloader_train)}")
    # print(f"eval: {len(dataloader_eval)}")

    print(f'length of eval dataloader: {len(dataloader_eval)}')

    train(
        model=model,
        # tokenizer=tokenizer,
        dataloader_train=dataloader_train,
        # eval_df = eval_df,
        dataloader_eval=dataloader_eval,
        device=device,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay_rate,
        num_warmup_steps=args.num_warmup_steps,
        accumulation_steps=args.gradient_accumulation_steps,
        eval_every=args.eval_every,
        patience=args.patience,
        best_model_dir= args.best_model_dir,
    )

