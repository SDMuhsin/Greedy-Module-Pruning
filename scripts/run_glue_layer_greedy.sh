#!/bin/bash

# Define arrays for model names, seeds, and task names
models=("bert-base-uncased")
seeds=(40 41 42 43 44)
tasks=("mrpc")
prune_layers=(4)

# Loop through all combinations
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    for task in "${tasks[@]}"; do
      for layers in "${prune_layers[@]}"; do
        output_dir="experiments/tmp/${model}_${task}_seed${seed}_prune${layers}"
        
        python3 ./source/run_glue.py \
          --output_dir "$output_dir" \
          --overwrite_output_dir \
          --model_name_or_path "$model" \
          --seed "$seed" \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --num_train_epochs 3 \
          --learning_rate 2e-5 \
          --task_name "$task" \
          --prune_n_layers "$layers" \
          --prune_method "prune-greedy" \
	 --job_id "prune-greedy-baselines" 
      done
    done
  done
done
