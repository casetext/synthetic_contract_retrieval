#!/bin/bash

train_folder="cuad_data_different_clauses"
train_file="train_cuad_cross_encoder"

for i in 1 2 3 4 5
do
    python train.py --model_name_or_path "nlpaueb/legal-bert-base-uncased" --train_path "../../data/${train_folder}/${train_file}.csv" --best_model_dir "./cross_encoder_${train_folder}_${i}"
    for csv in "test_cuad_data_different_clauses" "test_applicaai_v1" "test_applicaai_v2" "test_cuad_data_same_clauses"
    do
        python evaluate_cross_encoder.py --model_name_or_path "./cross_encoder_${train_folder}_${i}" --data_path "../../data/test_sets/${csv}.csv" --results_save_path "cross_encoder_${train_folder}_${csv}_${i}"
    done
done
exit 0
