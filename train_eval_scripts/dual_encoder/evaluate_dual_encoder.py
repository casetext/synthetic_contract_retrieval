import numpy as np
import pandas as pd

import json
import jsonlines

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from model import AutoDualEncoder

import argparse
import os
import pprint
import ipdb

import pickle

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
    all_queries,
    all_results,
    device,
    max_length=128,
    results_batch_size=64
):

    tokenized_query_sentences = tokenizer.batch_encode_plus(
                                all_queries, 
                                add_special_tokens=True,
                                return_attention_mask=True,
                                padding=True,
                                truncation=True,
                                max_length=max_length,
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
                            max_length=max_length,
                            return_tensors='pt'
    )

    positive_match_dataset = TensorDataset(
        tokenized_positive_matches["input_ids"],
        tokenized_positive_matches["attention_mask"]
        
    )

    positive_match_dataloader = DataLoader(
        positive_match_dataset,
        sampler=SequentialSampler(positive_match_dataset),
        batch_size=results_batch_size       
    )

    all_query_similarities = []
    
    progress_bar = tqdm(query_dataloader, leave=False, disable=False)

    for j, query in enumerate(progress_bar):
        
        similarities = []
        counter = 0
        for i, positive_match in enumerate(positive_match_dataloader):
            with torch.no_grad():

                forward_pool_query = model.forward_pool_query(query[0].to(device), query[1].to(device))
                forward_pool_passage = model.forward_pool_passage(positive_match[0].to(device), positive_match[1].to(device))
                forward_pool_query_norm = torch.linalg.norm(forward_pool_query, dim=1)
                forward_pool_passage_norm = torch.linalg.norm(forward_pool_passage, dim=1)

                normalized_forward_pool_query = forward_pool_query / forward_pool_query_norm[0]
                normalized_forward_pool_passage = forward_pool_passage / forward_pool_passage_norm.unsqueeze(1)

                ## batched processing ###               
                # [1, 1, emb_dim]
                normalized_forward_pool_query = normalized_forward_pool_query.unsqueeze(1)
                normalized_forward_pool_query_shape = normalized_forward_pool_query.shape

                # [bs, 1, emb_dim]
                broadcasted_normalized_forward_pool_query = torch.broadcast_to(
                        normalized_forward_pool_query, 
                        (normalized_forward_pool_passage.shape[0], normalized_forward_pool_query_shape[1], normalized_forward_pool_query_shape[2])
                )

                # [bs, emb_dim, 1]
                normalized_forward_pool_passage = normalized_forward_pool_passage.unsqueeze(2)

                batched_similarities = torch.bmm(
                    broadcasted_normalized_forward_pool_query,
                    normalized_forward_pool_passage
                )
                
                batched_similarities = batched_similarities.squeeze().squeeze().cpu().numpy()
                for similarity in batched_similarities:
                    similarities.append((counter, similarity))
                    counter += 1

        query_similarities_dict = {
            all_queries[j]: similarities
        }

        all_query_similarities.append(query_similarities_dict)
    
    return all_query_similarities

def compute_micro_precision_at_micro_recall(
    all_query_similarities,
    df, 
    all_results,
    output_destination,

):

    micro_average_recall_thresholds = [0.9, 0.95, 0.97]

    print('computing ground truths')
    query_to_relevant_ground_truth_dict = query_to_relevant_ground_truth(df)

    print('computing all query similarities')

    results_dict = {}
    
    for micro_average_recall_threshold in micro_average_recall_thresholds:
        print(f'Computing micro_average_precision and micro_average_portion_dataset at at {micro_average_recall_threshold}  recall ...')
        for threshold in np.arange(1.0, 0.0, -0.0001):

            total_results = 0
            micro_average_recall = 0.0
            micro_average_precision = 0.0
            micro_average_portion_dataset = 0.0

            macro_average_recall = 0.0
            macro_average_precision = 0.0
            macro_average_portion_dataset = 0.0

            all_data = []

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
                
                ## clause-wise performance
                data = {
                    "query": query,
                    "recall": recall,
                    "precision": precision,
                    "portion_dataset": frac_retrieved_dataset
                }
                all_data.append(data)
                ## clause-wise performance
                
                total_results += len(ground_truth_results)

                micro_average_recall += recall*len(ground_truth_results)
                micro_average_precision += precision*len(ground_truth_results)
                micro_average_portion_dataset += frac_retrieved_dataset*len(ground_truth_results)

                macro_average_recall += recall
                macro_average_precision += precision
                macro_average_portion_dataset += frac_retrieved_dataset

            micro_average_recall /= total_results
            micro_average_precision /= total_results
            micro_average_portion_dataset /= total_results

            macro_average_recall /= len(query_to_relevant_ground_truth_dict)
            macro_average_precision /= len(query_to_relevant_ground_truth_dict)
            macro_average_portion_dataset /= len(query_to_relevant_ground_truth_dict)

            if micro_average_recall >= micro_average_recall_threshold:
                print("threshold", threshold)
            # if macro_average_recall >= micro_average_recall_threshold:
                print("micro average precision", micro_average_precision)
                print("micro average portion dataset", micro_average_portion_dataset)
                print("micro average recall", micro_average_recall)

                # pprint.pprint(all_data)

                with open(f"clause_wise_{micro_average_recall_threshold}.pkl", "wb") as f:
                    pickle.dump(all_data, f)

                results_dict[micro_average_recall_threshold] = {
                            "micro_average_precision": micro_average_precision,
                            "micro_average_portion_dataset": micro_average_portion_dataset,
                            "micro_average_recall": micro_average_recall

                        }

                break
                
    if not os.path.exists("../../results_applica"):
        os.makedirs("../../results_applica")

    with jsonlines.open(f"../../results_applica/{output_destination}.jsonl", mode="w") as writer:
        for key, val in results_dict.items():
            writer.write(
                {key: val}
            )

    return results_dict
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing arguments')

    parser.add_argument('--model_name_or_path', metavar='N', type=str, default='bert-base-uncased',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--data_path', metavar='N', type=str, default='../../data/test_cuad_data.csv',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')
    parser.add_argument('--results_save_path', metavar='N', type=str, default='results',
                        help='Path where the tokenizer is saved, can be a huggingface model name as well')

    args, unknown = parser.parse_known_args()
 

    # Load data
    test_data_path = args.data_path
    test_data = pd.read_csv(test_data_path)

    all_queries = test_data["query"].unique().tolist()
    all_results = test_data["text"].unique().tolist()

    # Load model checkpoint
    question_load_path = f"./{args.model_name_or_path}/question_encoder"
    passage_load_path = f"./{args.model_name_or_path}/passage_encoder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoDualEncoder.from_pretrained(question_load_path)

    model = model.to(device)

    model.load_passage_encoder(passage_load_path)
    model.passage_encoder.to(device)
    model.eval()

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

    all_query_similarities = query_result_similarities(
        model=model,
        tokenizer=tokenizer,
        all_queries=all_queries,
        all_results=all_results,
        device=device
    )
    # ipdb.set_trace()

    results_dict = compute_micro_precision_at_micro_recall(
        all_query_similarities=all_query_similarities,
        df=test_data,
        all_results=all_results,
        output_destination=args.results_save_path
    )

