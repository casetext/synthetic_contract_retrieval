import torch
import numpy as np
import pandas as pd

from model import AutoDualEncoder
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

    tokenized_query_sentences = tokenizer.batch_encode_plus(
                                all_queries, 
                                add_special_tokens=True,
                                return_attention_mask=True,
                                padding='max_length',
                                truncation=True,
                                max_length=128,
                                return_tensors='pt'
    )

    query_dataset = TensorDataset(
                                tokenized_query_sentences["input_ids"],
                                tokenized_query_sentences["attention_mask"],
        
    )

    query_dataloader = DataLoader(
                                query_dataset,
                                sampler=SequentialSampler(query_dataset),
                                batch_size=1
    )

    tokenized_positive_matches = tokenizer.batch_encode_plus(
                                all_results,
                                add_special_tokens=True,
                                return_attention_mask=True,
                                padding='max_length',
                                truncation=True,
                                max_length=128,
                                return_tensors='pt'
    )

    positive_match_dataset = TensorDataset(
        tokenized_positive_matches["input_ids"],
        tokenized_positive_matches["attention_mask"]
        
    )

    positive_match_dataloader = DataLoader(
        positive_match_dataset,
        sampler=SequentialSampler(positive_match_dataset),
        batch_size=1
    )

    for query in all_queries:
        for result in all_results:
            queries.append(query)
            results.append(result)
            if result in query_to_relevant_gt[query]:
                relevance.append(1)
            else:
                relevance.append(0)
    
    similarities = []
    for j, query in enumerate(query_dataloader):

        for i, positive_match in tqdm(enumerate(positive_match_dataloader)):
            with torch.no_grad():

                forward_pool_query = model.forward_pool_query(query[0].to(device), query[1].to(device))
                forward_pool_passage = model.forward_pool_passage(positive_match[0].to(device), positive_match[1].to(device))

                forward_pool_query_norm = torch.linalg.norm(forward_pool_query, dim=1)
                forward_pool_passage_norm = torch.linalg.norm(forward_pool_passage, dim=1)

                normalized_forward_pool_query = forward_pool_query / forward_pool_query_norm[0]
                normalized_forward_pool_passage = forward_pool_passage / forward_pool_passage_norm[0]

                similarity = torch.bmm(normalized_forward_pool_query.unsqueeze(1), normalized_forward_pool_passage.unsqueeze(2)).squeeze().squeeze().cpu().item()

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
    question_load_path = f"./{args.model_name_or_path}/question_encoder"
    passage_load_path = f"./{args.model_name_or_path}/passage_encoder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoDualEncoder.from_pretrained(question_load_path)

    model = model.to(device)

    model.load_passage_encoder(passage_load_path)
    model.passage_encoder.to(device)
    model.eval()

    csv = compute_violin_plot_scores(   model,
                                        tokenizer,
                                        query_to_relevant_ground_truth_dict, 
                                        all_queries, 
                                        all_results,
                                        device
                                    )

    csv.to_csv(args.results_save_path)




