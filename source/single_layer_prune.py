#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
from models.bert.modeling_bert import LowRankBertLayer
import logging
import copy
import os
import random
import sys
import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# Custom pruning models
from models.model_factory import create_model
from disable_checkpoint_handler import DisableCheckpointCallbackHandler

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


# See https://github1s.com/huggingface/transformers/blob/HEAD/src/transformers/training_args.py
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    prune_n_layers: int = field(
        default=6,
        metadata={
            "help": "The maximum number of layers that is pruned afterwards. "
        },
    )
    prune_method: Optional[str] = field(
        default="greedy", metadata={"help": "Prune greedy in O(n^2) or perfect in O(n!)."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():

    if not os.path.exists("experiments/layer_files"):
        os.makedirs("experiments/layer_files")

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    #training_args.output_dir = f"{training_args.output_dir}/{data_args.task_name}/{model_args.model_name_or_path}/{model_args.prune_method}/{str(model_args.prune_n_layers)}/{str(training_args.seed)}"
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, cache_dir=".cache/")
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]

    # If the following line is uncommented, the dev set is used. Otherwise a 15% split of the training set is used.
    # eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    train_size = int(len(train_dataset) * 0.85)
    split_dataset = train_dataset.train_test_split(train_size=train_size, shuffle=True)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Create pruning strategy
    #prune_fn = greedy_module_prune if data_args.prune_method == "greedy-module" else prune_optimal
    #prune_fn(config, data_args, model_args, training_args, train_dataset,
    #        eval_dataset, compute_metrics, tokenizer, data_collator, datasets)

    single_layer_prune(config, data_args, model_args, training_args, train_dataset,
            eval_dataset, compute_metrics, tokenizer, data_collator, datasets)

def single_layer_prune(config, data_args, model_args, training_args, train_dataset,
    eval_dataset, compute_metrics, tokenizer, data_collator, datasets):

    ''' Step 1 : Gather list of layers '''
    model = create_model(config,model_args)
    layers = range(config.num_hidden_layers)


    ''' Step 2 : Iterate through layers exhaustively'''
    scores_per_layer = {}
    for layer_idx in layers:

        layers_to_prune = [l for l in range( config.num_hidden_layers ) if l != layer_idx]
        print(f"Pruning all but {layer_idx} : {layers_to_prune}")
        
        ''' Step 3 : For each layer, make a model consisting of that layer only '''
        model_under_test = copy.deepcopy(model)
        model_under_test.prune_layers(layers_to_prune)



        ''' Step 4 : Train this single layer model and evaluate '''
        res = just_evaluate_model(model, config, model_args, data_args,
                    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator, datasets) 

        print(f"Layer {layer_idx} scored : {res}")
        scores_per_layer[layer_idx] = res

    ''' Step 5 : Save the list of layers to prune'''
    
    # The top layers are the best layers to prune, so least important layers should be on top, so store layers in ascending order
    sorted_layers = sorted(scores_per_layer, key=scores_per_layer.get)
    print("Layers ordered by worst to best performance {sorted_layers}")

    with open(f'./layer_files/{model_args.model_name_or_path}_{data_args.task_name}_single_layer.txt', 'w') as f:
        f.writelines("%s\n" % l for l in sorted_layers)
    return None


def greedy_module_prune(config, data_args, model_args, training_args, train_dataset,
    eval_dataset, compute_metrics, tokenizer, data_collator, datasets):

    ''' Step 1 : Construct pruning state '''
    model = create_model(config,model_args)
    pruning_state = generate_layer_weights_dict(model)



    ''' Step 2 : Iterate through layers greedily'''

    print(pruning_state)
    num_iterations_left = min(2 * data_args.prune_n_layers, config.num_hidden_layers-1) # NOTE the 2x because we need to prune twice as many layers by half to match full layer pruning
    
    pruned_layers = []
    while num_iterations_left:
        
        # Go through available layers
        available_layers = list( set(range(0,config.num_hidden_layers)) - set(pruned_layers) )

        current_best_layer_to_prune, current_best_score = -1,-100
        for layer_idx in available_layers:
            
            # Prune this layer
                # Update temp pruning_state
                # Prune model
            temp_pruning_state = copy.deepcopy(pruning_state)
            temp_pruning_state = update_pruning_state_prune_layer(temp_pruning_state,layer_idx)
            
            print_pruning_state(temp_pruning_state)
            
            res = prune_and_evaluate_model(temp_pruning_state, config, model_args, data_args,
                    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator, datasets)
            print("RESULT : ",res)
            exit()
            # Fine tune and evaluate

            # Record performance

        

        pruned_layers.append(available_layers[0])
        
        num_iterations_left -= 1

    return None
def just_evaluate_model(model, config, model_args, data_args,
    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator,
    datasets):

    # Set seed before initializing model.
    set_seed(training_args.seed) 
    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.add_callback(
        DisableCheckpointCallbackHandler()
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics

    # Evaluation
    eval_results = {}
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_mismatch = datasets["validation_mismatched"]
        eval_datasets.append(eval_mismatch)

    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        eval_results.update(eval_result)

    # res = eval_results.get("eval_loss", None)
    res = None
    res = res or eval_results.get("eval_f1", None)
    res = res or eval_results.get("eval_accuracy", None)
    res = res or eval_results.get("eval_spearmanr", None)
    res = res or eval_results.get("eval_matthews_correlation", None)

    res = round(res, 3)

    if(res == None):
        raise Exception("Now performance metric found!")

    return res
def prune_and_evaluate_model(pruning_state, config, model_args, data_args,
    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator,
    datasets):

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = create_model(config, model_args)
    prune_model_modules(model,pruning_state)
    
    ''' Adjust dim params : For each layer, Make a version with reduced dimensions and copy the weights over '''

    print(config)
    for layer_idx in range(config.num_hidden_layers):
        
        # Identify hidden_size and intermediate size for this layer
        #   Get weight names for this layer
        layer_weights_names = get_layer_weights(pruning_state,layer_idx)

        #   Strat for MHA and FFN

        mha_weight = get_matching_weights(layer_weights_names,"self.query.weight")
        ffn_weight = get_matching_weights(layer_weights_names,"intermediate.dense.weight")

        assert len(mha_weight) == 1 , f"{mha_weight}"
        assert len(ffn_weight) == 1, f"{ffn_weight}"

        mha_strat = pruning_state[ mha_weight[0] ].split("-")[-1]
        ffn_strat = pruning_state[ ffn_weight[0] ].split("-")[-1]
        

        if(mha_strat.isnumeric() or ffn_strat.isnumeric()):
            
            print(f"Copying over weights")
            print(f"MHA Compression = {mha_strat}")
            mha_compression = int(mha_strat) if mha_strat.isnumeric() else 1
            ffn_compression = int(ffn_strat) if ffn_strat.isnumeric() else 1

            new_bert_layer = LowRankBertLayer(config, mha_compression, ffn_compression)
            transfer_layer_weights(model,new_bert_layer,layer_idx)

            print(f"Copied over weights for layer {layer_idx}")
    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.add_callback(
        DisableCheckpointCallbackHandler()
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics

    # Evaluation
    eval_results = {}
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_mismatch = datasets["validation_mismatched"]
        eval_datasets.append(eval_mismatch)

    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        eval_results.update(eval_result)

    # res = eval_results.get("eval_loss", None)
    res = None
    res = res or eval_results.get("eval_f1", None)
    res = res or eval_results.get("eval_accuracy", None)
    res = res or eval_results.get("eval_spearmanr", None)
    res = res or eval_results.get("eval_matthews_correlation", None)

    res = round(res, 3)

    if(res == None):
        raise Exception("Now performance metric found!")

    return res
def transfer_layer_weights(model, new_bert_layer, layer_idx):
    # Access the specific layer in the full BERT model
    source_layer = model.bert.encoder.layer[layer_idx]

    # Iterate over the parameters of the source and target layers
    source_params = list(source_layer.parameters())
    target_params = list(new_bert_layer.parameters())

    assert len(source_params) == len(target_params), "Number of parameters do not match between source and target layers"

    for source_param, target_param in zip(source_params, target_params):
        assert source_param.size() == target_param.size(), f"Parameter size mismatch: {source_param.size()} vs {target_param.size()}"
        target_param.data.copy_(source_param.data)

def get_matching_weights(weights,match_id):
    return [weight for weight in weights if match_id in weight]

def prune_model_modules(model,model_state):
    
    for weight_name,pruning_strat in model_state.items():
        
        if(pruning_strat == "NOP"):
            continue
        
        #print(f"Pruning weight : {weight_name}")

        param_matrix = get_weight_by_name(model,weight_name)
        #print(f"Before {param_matrix.shape}")
        pruned_param_matrix = prune_param_matrix(param_matrix,pruning_strat)
        #print(f"After {pruned_param_matrix.shape}")
        set_weight_by_name(model, weight_name, pruned_param_matrix)

def set_weight_by_name(model, weight_name, new_weight):
    # Split the weight name into its components
    name_parts = weight_name.split('.')
    
    # Navigate through the model's modules and sub-modules
    module = model
    for part in name_parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    # Get the current parameter
    current_param = getattr(module, name_parts[-1])
    
    # Create a new parameter with the same properties as the current one
    new_param = torch.nn.Parameter(new_weight.to(current_param.device), 
                                   requires_grad=current_param.requires_grad)
    
    # Set the new parameter
    setattr(module, name_parts[-1], new_param)
        
def prune_param_matrix(param_matrix, pruning_strat):
    # Split the pruning strategy into its components
    parts = pruning_strat.split('-')
    if len(parts) != 3:
        raise ValueError("Invalid pruning strategy format. Expected format: <C/R>-<F/L>-<N>")
    
    prune_type, prune_position, prune_fraction = parts
    prune_fraction = int(prune_fraction)

    if prune_type not in ['C', 'R']:
        raise ValueError("Invalid prune type. Expected 'C' for columns or 'R' for rows.")
    if prune_position not in ['F', 'L']:
        raise ValueError("Invalid prune position. Expected 'F' for first or 'L' for last.")

    # Handle 1D tensors (biases)
    if param_matrix.dim() == 1:
        num_elements = param_matrix.shape[0]
        prune_count = num_elements // prune_fraction
        if prune_position == 'F':
            pruned_param_matrix = param_matrix[prune_count:]
        elif prune_position == 'L':
            pruned_param_matrix = param_matrix[:-prune_count]
    # Handle 2D tensors (weights)
    elif param_matrix.dim() == 2:
        if prune_type == 'C':
            num_columns = param_matrix.shape[1]
            prune_count = num_columns // prune_fraction
            if prune_position == 'F':
                pruned_param_matrix = param_matrix[:, prune_count:]
            elif prune_position == 'L':
                pruned_param_matrix = param_matrix[:, :-prune_count]
        elif prune_type == 'R':
            num_rows = param_matrix.shape[0]
            prune_count = num_rows // prune_fraction
            if prune_position == 'F':
                pruned_param_matrix = param_matrix[prune_count:, :]
            elif prune_position == 'L':
                pruned_param_matrix = param_matrix[:-prune_count, :]
    else:
        raise ValueError("Unsupported tensor dimension. Expected 1D or 2D tensor.")

    return pruned_param_matrix

def update_pruning_state_prune_layer(pruning_state,layer_idx):

    layer_weights = get_layer_weights(pruning_state,layer_idx)
    for layer_weight in layer_weights:
            
        # Branch based on what kind of layer weight matrix it is

        # qwery, key, value : C-L-2
        if ( match_weight_name(layer_weight, ['self.key','self.query','self.value','intermediate.dense'] )):
            pruning_state[layer_weight] = 'C-L-2'
        # Output dense R-L-2
        elif( match_weight_name(layer_weight,['attention.output.dense','output.dense'])):
            pruning_state[layer_weight] = 'R-L-2'
    return pruning_state

def print_pruning_state(pruning_state):
    
    print(" --- Pruned configs ---")
    for k,v in pruning_state.items():
        if(v != 'NOP'):
            print(f"{k} : {v}")
    print(" --- Pruned configs ---")

def match_weight_name(layer_weight: str, name_list: list[str]) -> bool:
    """
    Check if any of the strings in name_list are present in layer_weight.

    Args:
    layer_weight (str): The name of the layer weight to check.
    name_list (list[str]): A list of strings to match against.

    Returns:
    bool: True if any string in name_list is found in layer_weight, False otherwise.
    """
    return any(name in layer_weight for name in name_list)


def generate_layer_weights_dict(module: nn.Module) -> dict:

    weight_dict = {}

    def get_full_name(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            for param_name, param in child.named_parameters(recurse=False):
                if param.requires_grad and "layer" in full_name:
                    weight_dict[f"{full_name}.{param_name}"] = "NOP"
            get_full_name(child, full_name)

    get_full_name(module)
    return weight_dict

def get_layer_weights(pruning_state,layer_idx):
    
    layer_weights = []
    for k,v in pruning_state.items():
        if(f"layer.{layer_idx}." in k):
            layer_weights.append(k)
    return layer_weights

def prune_optimal(config, data_args, model_args, training_args, train_dataset,
    eval_dataset, compute_metrics, tokenizer, data_collator, datasets):
    num_layers = data_args.prune_n_layers
    all_layers = [i for i in range(config.num_hidden_layers)]
    file_rows = []

    for num_layers_to_prune in range(1, data_args.prune_n_layers+1):
        all_combinations = [i for i in itertools.combinations(all_layers, num_layers_to_prune)]
        cache_dict = {}

        print("\n#\n# TRAINING %d NETWORKS!\n#" % len(all_combinations))
        for pruned_layers in all_combinations:
            pruning_id = "-".join(map(str, pruned_layers))
            loss = evaluate_model(cache_dict, pruning_id, pruned_layers, config, model_args, data_args,
                    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator, datasets)

            # Log result for later analysis
            with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_perfect_log.txt', 'a') as f:
                f.writelines(f"{pruning_id};{loss[pruning_id]}\n")

        layer_to_prune = None
        for key in cache_dict.keys():
            if(layer_to_prune == None or cache_dict[key] >= cache_dict[layer_to_prune]):
                layer_to_prune = key
        file_rows.append(layer_to_prune)


    # DONE - store into file
    with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_perfect.txt', 'w') as f:
        f.writelines("%s\n" % l for l in file_rows)



def prune_greedy(config, data_args, model_args, training_args, train_dataset,
    eval_dataset, compute_metrics, tokenizer, data_collator, datasets):

    finally_pruned_layers = []
    cache_dict = {}
    num_iterations = min(data_args.prune_n_layers, config.num_hidden_layers-1)
    while(len(finally_pruned_layers) < num_iterations):
        layer_ids = [i for i in range(config.num_hidden_layers) if i not in finally_pruned_layers]
        lower_layer = 0
        upper_layer = len(layer_ids)-1
        middle_layer = int(upper_layer / 2)

        for i in range(0, len(layer_ids)):
            print("\n####")
            cache_dict = evaluate_model(cache_dict, layer_ids[i], finally_pruned_layers, config, model_args, data_args,
                training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator, datasets)
            print(cache_dict)

        # Log result for later analysis
        with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_greedy_log.txt', 'a') as f:
            f.writelines(f"{str(len(finally_pruned_layers))};{str(cache_dict)}\n")

        layer_to_prune = -1
        for key in cache_dict.keys():
            if(layer_to_prune < 0 or cache_dict[key] >= cache_dict[layer_to_prune]):
                layer_to_prune = key

        cache_dict = {}
        finally_pruned_layers.append(layer_to_prune)
        print("PRUNED LAYER %s" % finally_pruned_layers)

    # DONE - store into file
    with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_greedy.txt', 'w') as f:
        f.writelines("%s\n" % l for l in finally_pruned_layers)


def display_model_weight_matrices(model):
    for name, param in model.named_parameters():
        if param.dim() == 2:  # Check if the parameter is a 2D tensor (matrix)
            print(f"Name: {name} | Shape: {param.shape}")
            #print(f"Data:\n{param.data}")
            #print("-" * 50)

def get_weight_by_name(model, weight_name):
    # Split the weight name into its components
    attribute_hierarchy = weight_name.split('.')
    
    # Start with the model itself
    current_attr = model
    
    # Iterate through the hierarchy to get to the desired attribute
    for attr in attribute_hierarchy:
        current_attr = getattr(current_attr, attr)
    
    return current_attr
def evaluate_model(cache_dict, layer_id, finally_pruned_layers, config, model_args, data_args,
    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator,
    datasets):

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if(layer_id in cache_dict):
        print("Layer %d from cache: %.4f" % (layer_id, cache_dict[layer_id]))
        return cache_dict[layer_id]

    print(f"Calculate layer {str(layer_id)}")
    model = create_model(config, model_args)
    display_model_weight_matrices(model)

    model.prune_layers(finally_pruned_layers)
    if isinstance(layer_id, int):
        model.prune_layers([layer_id])

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.add_callback(
        DisableCheckpointCallbackHandler()
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics

    # Evaluation
    eval_results = {}
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_mismatch = datasets["validation_mismatched"]
        eval_datasets.append(eval_mismatch)

    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        eval_results.update(eval_result)

    # res = eval_results.get("eval_loss", None)
    res = None
    res = res or eval_results.get("eval_f1", None)
    res = res or eval_results.get("eval_accuracy", None)
    res = res or eval_results.get("eval_spearmanr", None)
    res = res or eval_results.get("eval_matthews_correlation", None)

    res = round(res, 3)

    if(res == None):
        raise Exception("Now performance metric found!")

    cache_dict[layer_id] = res
    return cache_dict


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()



if __name__ == "__main__":
    main()
