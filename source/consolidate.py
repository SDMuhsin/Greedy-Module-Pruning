import os
import json
from tabulate import tabulate
import statistics

# Specify the variable parameters
job_ids = ["prune-greedy-baselines","single-layer"]
prune_n_layers_list = [2,4,6]
tasks = ["rte", "mrpc", "stsb", "sst2", "cola", "qnli","mnli","qqp"]
models = ["bert-base-uncased", "roberta-base"]

# Specify the base directory
base_dir = "./saves"

# Task to metrics mapping
task_to_metrics = {
    "boolq": "accuracy",
    "cb": "accuracy",
    "wic": "accuracy",
    "wsc": "accuracy",
    "copa": "accuracy",
    "rte": "accuracy",
    "mrpc": "accuracy",
    "stsb": "pearson",
    "sst2": "accuracy",
    "cola": "matthews_correlation",
    "qnli": "accuracy"
}

# Function to read and process JSON files
def process_json_file(file_path, task):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        metric = f"eval_{task_to_metrics[task]}"
        
        # Collect scores for all available seeds
        scores = [run_data[metric] for run_data in data.values() if metric in run_data]
        
        # If we don't have data for all 5 seeds, return '-'
        if len(scores) < 5:
            return '-'
        
        # Sort scores and take the median (middle) value
        sorted_scores = sorted(scores)
        median_score = sorted_scores[len(sorted_scores) // 2]
        
        return f"{median_score:.4f}"
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return '-'

# Prepare the table data
table_data = []
for model in models:
    for prune_n_layers in prune_n_layers_list:
        for job_id in job_ids:
            row = [f"{model}-{prune_n_layers}-{job_id}"]
            for task in tasks:
                file_name = f"results_rg_ensemble_{task}_{model}.json"
                file_path = os.path.join(base_dir, f"{job_id}-{prune_n_layers}", file_name)
                
                cell_value = process_json_file(file_path, task)
                row.append(cell_value)
            table_data.append(row)

# Create the table
headers = ["Model-Prune-JobID"] + tasks
table = tabulate(table_data, headers=headers, tablefmt="grid")

# Print the table
print(table)
