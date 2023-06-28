# synthetic_contract_retrieval
Code for Gen-IR@SIGIR23 Paper "GPT-4 Synthetic Data Improves Generalizability For Contract Clause Retrieval"

# To generate data:

- Create a file named `data_generation/credentials.json` with your OpenAI `key` and `organization`
- Run `data_generation/synthetic_datagen.ipynb` to generate synthetic datasets from GPT-4
- Run `data_generation/preprocess_cuad_ds.ipynb` to create the baseline CUAD train and test sets both the cross and dual encoders.
- Run `data_generation/preprocess_synthetic_datasets.ipynb` to create the synthetic datasets (synthetic, synthetic discriminative and SQUAD).

# Training and evaluating a model

To train and evaluate an individual model, run:

- `sh ./train_eval_scripts/cross_encoder/train.sh` for a cross encoder
- `sh ./train_eval_scripts/dual_encoder/train.sh` for a dual encoder

Modify the `train_folder` and `train_file` as appropriate in the `data` directory. The scripts are default to run five times, this can be modified in the `for` loop. Once training and evaluation is done, the model will be saved in the directory of the train script while the metrics will be saved as `.jsonl` file in a `./results` folder in the top level directory of the repository. The results should look something like this (example):

```
{"0.9": {"micro_average_precision": 0.12920440829722774, "micro_average_portion_dataset": 0.41747621452929423, "micro_average_recall": 0.9001152516327314}}
{"0.95": {"micro_average_precision": 0.10629680006204627, "micro_average_portion_dataset": 0.5228543869134854, "micro_average_recall": 0.9500576258163658}}
{"0.97": {"micro_average_precision": 0.09646806820065935, "micro_average_portion_dataset": 0.5821436151544038, "micro_average_recall": 0.9700345754898194}}
```

