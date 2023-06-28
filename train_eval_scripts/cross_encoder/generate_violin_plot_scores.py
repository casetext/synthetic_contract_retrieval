import torch
import numpy as np
import pandas as pd

from model import AutoCrossEncoder
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import json
import jsonlines
import os, sys

import argparse

import ipdb

def query_to_relevant_ground_truth(df):

    df = df[["query", "text"]].drop_duplicates()
    relevant_ground_truths = df.groupby(["query"])["text"].apply(list)
    unique_queries = list(relevant_ground_truths.index)
    print (f"num unique queries in eval set: {len(unique_queries)}")

    query_to_relevant_gt = {}

    for i, query in enumerate(unique_queries):
        query_to_relevant_gt[query] = relevant_ground_truths[i]

    return query_to_relevant_gt

def compute_violin_plot_scores(model, tokenizer, query_to_relevant_gt, all_queries, all_results, device):

    queries = []
    results = []
    relevance = []

    # get the queries and their results in
    for query in all_queries:
        for result in all_results:
            queries.append(query)
            results.append(result)
            if result in query_to_relevant_gt[query]:
                relevance.append(1)
            else:
                relevance.append(0)

    
    similarities = []

    for query in tqdm(all_queries):

        all_pairs = [[query,result] for result in all_results]

        encoded_pairs = tokenizer.batch_encode_plus(
                                    all_pairs,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    return_token_type_ids=True,
                                    padding=True,
                                    truncation=True,
                                    max_length=256,
                                    return_tensors='pt'
                                    )

        dataset = TensorDataset(
                                encoded_pairs["input_ids"],
                                encoded_pairs["attention_mask"],
                                encoded_pairs["token_type_ids"]
        )

        dataloader = DataLoader(
                                dataset,
                                sampler=SequentialSampler(dataset),
                                batch_size=1
        )

        for i, input_pairs in tqdm(enumerate(dataloader)):
            with torch.no_grad():
            
                similarity = model.forward(
                    input_ids=input_pairs[0].to(device),
                    attention_mask=input_pairs[1].to(device),
                    token_type_ids=input_pairs[2].to(device)
                )

                similarity = similarity.cpu().item()
                similarities.append(similarity)

    csv = pd.DataFrame({
        "query": queries,
        "results": results,
        "relevance": relevance,
        "score": similarities
    })

    return csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing arguments')

    parser.add_argument('--model_name_or_path', metavar='N', type=str, default='bert-base-uncased',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--data_path', metavar='N', type=str, default='../../data/test_cuad_data.csv',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--results_save_path', metavar='N', type=str, default='test.csv',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')

    args, unknown = parser.parse_known_args()

    # Load data
    test_data_path = args.data_path
    test_data = pd.read_csv(test_data_path)

    all_queries = test_data["query"].unique().tolist()
    all_results = test_data["text"].unique().tolist()

    print(f"number of unique queries: {len(all_queries)}")
    print(f"number of unique clauses: {len(all_results)}")

    query_to_relevant_ground_truth_dict = query_to_relevant_ground_truth(test_data)

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoCrossEncoder(f"./{args.model_name_or_path}/crossencoder")
    model = model.to(device)
    model.eval()

    csv = compute_violin_plot_scores(   model,
                                        tokenizer,
                                        query_to_relevant_ground_truth_dict, 
                                        all_queries, 
                                        all_results,
                                        device
                                    )
    
    csv.to_csv(args.results_save_path)

    







