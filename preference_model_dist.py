from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import collections
import argparse
import logging
import pandas as pd
import tqdm
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, dataset
import torch.nn.functional as F
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5PreTrainedModel, T5EncoderModel, T5Model, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput
from transformers.trainer_utils import seed_worker
from datasets import load_dataset
import datasets
import json
import random
from scipy.stats import pearsonr, spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DataCollatorForData2TextLanguageModeling:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, examples: Dict[str, str]) -> Dict[str, torch.Tensor]:
        # input_text, output_text = tuple([example[i] for example in examples] for i in (0,1))
        input_text = [ example['x'] for example in examples ]
        output_text = [ str(example['y']) for example in examples ]

        inputs = self.tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = self.tokenizer(output_text, max_length=2, truncation=True, return_tensors="pt")
        return dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=outputs.input_ids)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-xl")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="/nlp/scr/kawin/.cache")


def train(data_prefix='combined_'):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # Ampere only.

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns=False
    print("Will write to...", training_args.output_dir)

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
    )

    print("Loading dataset...")
    data_files = {
            "train": f"data/{data_prefix}train.json", 
            "validation": f"data/{data_prefix}test.json"
            }
    logging.warning("starting data load")
    data = load_dataset("json", data_files=data_files)
    logging.warning("Done loading data!")

    data_module = dict(
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        data_collator=DataCollatorForData2TextLanguageModeling(tokenizer=tokenizer),
    )

    trainer = transformers.Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


def eval(model_fn, data_prefix="combined_"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_fn).to(device)
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_fn)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"test": f"data/{data_prefix}test.json"})['test']
    test_dataloader = DataLoader(dataset, batch_size=8)

    progress_bar = tqdm.tqdm(test_dataloader)
    total, num_correct = 0, 0
    domain_correct, domain_total = {}, {}
    pvi_values = []
    loss_curve = []

    for batch in progress_bar: 
        input_ids = tokenizer(batch["x"], padding=True, truncation=True, max_length=500, return_tensors="pt").input_ids.to(device)  
        outputs = model.generate(input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        total += len(predictions)
        num_correct += (np.array(batch["y"], dtype=str) == predictions).sum()

        acc = num_correct / total
        progress_bar.set_description(f'Accuracy: {acc:.04f}')
        
        # ablation study
        for i in range(len(batch["y"])):
            d = batch["domain"][i]

            if d not in domain_correct:
                domain_correct[d] = 0
                domain_total[d] = 0

            domain_correct[d] += (str(batch["y"][i]) == predictions[i])
            domain_total[d] += 1

            # token IDs are 272 for B when "<pad> _B" is generated and 71 for A when "<pad> _A" is generated
            label_idx = 272 if batch["y"][i] == "B" else 71
            p_yx = np.exp(outputs.scores[0][i, label_idx].item()) / np.exp(outputs.scores[0][i,:].cpu().numpy()).sum()
            p_y = dataset["y"].count(batch["y"][i]) / len(dataset["y"])
            # calculate conditional v-entropy in bits (base 2)
            pvi = -np.log2(p_y) + np.log2(p_yx)
            pvi_values.append(pvi)

            loss_curve.append({ 'score_ratio': batch['score_ratio'][i].item(), 'correct': (str(batch["y"][i]) == predictions[i]) })

    for d in domain_correct:
        print(d, domain_correct[d] / domain_total[d])

    print('Average Accuracy:', acc)
    print('V-info Estimate', np.mean(pvi_values))

    return pd.DataFrame.from_dict(loss_curve)


def eval_regress(model_fn, data_prefix="combined_"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_fn).to(device)
    model.eval()
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_fn)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"test": f"data/{data_prefix}test.json"})['test']
    test_dataloader = DataLoader(dataset, batch_size=8)

    progress_bar = tqdm.tqdm(test_dataloader)
    total, num_correct = 0, 0
    domain_correct, domain_total = {}, {}
    pvi_values = []
    loss_curve = []

    def break_down_examples(batch):
        only_A, only_B = [], []

        for x in batch["x"]:
            x = re.split("\n\n ", x)
            only_B.append("\n\n ".join([ x[0], x[2].replace("RESPONSE B", "RESPONSE A"), "RESPONSE B: .", x[3] ]))
            only_A.append("\n\n ".join([ x[0], x[1], "RESPONSE B: .", x[3] ]))

        return only_A, only_B 

    for batch in progress_bar: 
        only_A, only_B = break_down_examples(batch)
        
        only_A_input_ids = tokenizer(only_A, padding=True, truncation=True, max_length=500, return_tensors="pt").input_ids.to(device)  
        only_A_outputs = model.generate(only_A_input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        p_A = torch.exp(only_A_outputs.scores[0][:, 71]) / torch.exp(only_A_outputs.scores[0][:,:]).sum(axis=1)

        only_B_input_ids = tokenizer(only_B, padding=True, truncation=True, max_length=500, return_tensors="pt").input_ids.to(device)  
        only_B_outputs = model.generate(only_B_input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        p_B = torch.exp(only_B_outputs.scores[0][:, 71]) / torch.exp(only_B_outputs.scores[0][:,:]).sum(axis=1)

        p_diff = (p_A - p_B)
        predictions = [ "A" if p_diff[i].item() > 0 else "B" for i in range(len(batch["y"])) ]
        predictions = np.array(predictions)
        
        total += len(predictions)
        num_correct += (np.array(batch["y"], dtype=str) == predictions).sum()

        acc = num_correct / total
        progress_bar.set_description(f'Accuracy: {acc:.04f}')
        
        # ablation study
        for i in range(len(batch["y"])):
            d = batch["domain"][i]

            if d not in domain_correct:
                domain_correct[d] = 0
                domain_total[d] = 0

            domain_correct[d] += (str(batch["y"][i]) == predictions[i])
            domain_total[d] += 1

            loss_curve.append({ 'score_ratio': batch['score_ratio'][i].item(), 'correct': (str(batch["y"][i]) == predictions[i]) })

    for d in domain_correct:
        print(d, domain_correct[d] / domain_total[d])

    print('Accuracy:', acc)
    return pd.DataFrame.from_dict(loss_curve)


if __name__ == "__main__":
    train()
    # eval("/nlp/scr/kawin/models/flan-t5-large-rerun")
