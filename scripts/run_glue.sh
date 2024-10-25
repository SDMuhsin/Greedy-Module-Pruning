#!/bin/bash

# Define arrays for model names, seeds, and task names
models=("bert-base-uncased")
seeds=(40 41 42 43 44)
tasks=("mrpc" "rte" "stsb" "cola")
prune_layers=(4)

# Generate the combinations and run them in parallel
parallel -j 2 -u python3 ./source/run_glue.py \
  --output_dir experiments/tmp/{1}_{3}_seed{2}_prune{4} \
  --overwrite_output_dir \
  --model_name_or_path {1} \
  --seed {2} \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --task_name {3} \
  --prune_n_layers {4} \
  --prune_method single-layer-prune \
  --job_id single-layer ::: "${models[@]}" ::: "${seeds[@]}" ::: "${tasks[@]}" ::: "${prune_layers[@]}"
