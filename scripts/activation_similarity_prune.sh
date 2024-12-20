#!/bin/bash

source env/bin/activate

#
# Prune greedy
#
for task in "rte" "mrpc" "stsb" "cola"; do
for model in  "roberta-base" "bert-base-uncased" ; do
python3 ./source/activation_similarity_prune.py --model_name_or_path=$model \
    --task_name=$task \
    --seed=41 \
    --max_seq_length=128 \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --output_dir=experiments/tmp/ \
    --logging_dir=experiments/tmp/ \
    --prune_n_layers=4 \
    --prune_method="activation_similarity" \
    --overwrite_output_dir
done
done

exit 1
#
# Prune optimal
#
for task in "sst2" "cola" "mnli" "mrpc" "qnli" "qqp" "rte" "stsb"; do
for model in "bert-base-uncased" "roberta-base"; do
python3 analyze.py --model_name_or_path=$model \
    --task_name=$task \
    --seed=41 \
    --max_seq_length=128 \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --output_dir=experiments/tmp/ \
    --logging_dir=experiments/tmp/ \
    --prune_n_layers=2 \
    --prune_method="optimal" \
    --overwrite_output_dir
done
