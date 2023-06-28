import numpy as np
import pandas as pd

import json
import jsonlines

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from model import AutoCrossEncoder

import argparse
import ipdb
import os

def query_to_relevant_ground_truth(df):

    df = df[["query", "text"]].drop_duplicates()
    relevant_ground_truths = df.groupby(["query"])["text"].apply(list)
    unique_queries = list(relevant_ground_truths.index)
    print (f"num unique queries in eval set: {len(unique_queries)}")

    query_to_relevant_gt = {}

    for i, query in enumerate(unique_queries):
        query_to_relevant_gt[query] = relevant_ground_truths[i]

    return query_to_relevant_gt

def query_result_similarities(
    model, 
    tokenizer, 
    df,
    device,
    batch_size=64
):

    all_queries = df['query'].unique().tolist()
    all_results = df['text'].unique().tolist()
    
    all_query_similarities = []
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
                                batch_size=batch_size
        )
       
        similarities = []
        counter = 0
        for i, input_pairs in enumerate(dataloader):
            with torch.no_grad():
            
                similarity = model.forward(
                    input_ids=input_pairs[0].to(device),
                    attention_mask=input_pairs[1].to(device),
                    token_type_ids=input_pairs[2].to(device)
                )

                similarity = similarity.cpu().numpy().astype('float')
                for s in similarity:
                    similarities.append((counter,s[0]))
                    counter += 1
            
        query_similarities_dict = {
            query: similarities
        }

        all_query_similarities.append(query_similarities_dict)
    
    return all_query_similarities

def compute_micro_precision_at_micro_recall(
    all_query_similarities, 
    df,
    output_destination,
):

    micro_average_recall_thresholds = [0.9, 0.95, 0.97]

    print('computing ground truths')
    query_to_relevant_ground_truth_dict = query_to_relevant_ground_truth(df)

    print('computing all query similarities')

    all_results = df['text'].unique().tolist()

    results_dict = {}

    for micro_average_recall_threshold in micro_average_recall_thresholds:
        print(f'Computing micro_averate_precision and micro_average_portion_dataset at at {micro_average_recall_threshold} percent recall ...')
        for threshold in tqdm(np.arange(1.0, 0.0, -0.00001)):

            total_results = 0
            micro_average_recall = 0.0
            micro_average_precision = 0.0
            micro_average_portion_dataset = 0.0

            for query_similarity_dict in all_query_similarities:

                query = list(query_similarity_dict.keys())[0]
                similarities = query_similarity_dict[query]
                thresholded_similarities = [el for el in similarities if el[1] > threshold]
                frac_retrieved_dataset = len(thresholded_similarities) / len(all_results)

                retrieved_passages = []
                for el in thresholded_similarities:
                    index = el[0]
                    retrieved_passages.append(all_results[index])

                ground_truth_results = query_to_relevant_ground_truth_dict[query]

                try:
                    recall = len(set(retrieved_passages).intersection(set(ground_truth_results))) / len(ground_truth_results)
                    precision = len(set(retrieved_passages).intersection(set(ground_truth_results))) / len(retrieved_passages)
                
                except ZeroDivisionError:
                    recall = 0.0
                    precision = 0.0
                
                total_results += len(ground_truth_results)
                
                micro_average_recall += recall*len(ground_truth_results)
                micro_average_precision += precision*len(ground_truth_results)
                micro_average_portion_dataset += frac_retrieved_dataset*len(ground_truth_results)

            micro_average_recall /= total_results
            micro_average_precision /= total_results
            micro_average_portion_dataset /= total_results

            if micro_average_recall >= micro_average_recall_threshold:
                print("micro average precision", micro_average_precision)
                print("micro average portion dataset", micro_average_portion_dataset)
                print("micro average recall", micro_average_recall)

                results_dict[micro_average_recall_threshold] = {
                            "micro_average_precision": micro_average_precision,
                            "micro_average_portion_dataset": micro_average_portion_dataset,
                            "micro_average_recall": micro_average_recall

                        }

                break
    
    if not os.path.exists('../../results/'):
        os.makedirs('../../results/')
    with jsonlines.open(f"../../results/{output_destination}.jsonl", mode="w") as writer:
        for key, val in results_dict.items():
            writer.write(
                {key: val}
            )

    return results_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data preprocessing arguments')

    parser.add_argument('--model_name_or_path', metavar='N', type=str, default='nlpaueb/legal-bert-base-uncased',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--data_path', metavar='N', type=str, default='../../data/test_cuad_data.csv',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--results_save_path', metavar='N', type=str, default='results',
                        help='filename in ../data/ where the results jsonl will be saved')
                        
    args, unknown = parser.parse_known_args()
    
    # Load data
    test_data = pd.read_csv(args.data_path)

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoCrossEncoder(f"./{args.model_name_or_path}/crossencoder")
    model = model.to(device)
    model.eval()

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

    all_query_similarities = query_result_similarities(
        model,
        tokenizer,
        test_data,
        device
    )

    results_dict = compute_micro_precision_at_micro_recall(
        all_query_similarities,
        test_data,
        args.results_save_path
    )

