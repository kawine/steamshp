import os
import json
import pandas as pd
import numpy as np
import pysbd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import T5Tokenizer
from abc import ABC, abstractmethod
from typing import List
from config import SUBREDDITS, ANTHROPIC


class SubSampler(ABC):
    """
    An abstract class for subsampling the training data.
    """
    @abstractmethod
    def subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RatioSampler(SubSampler):
    """
    A SubSampler for the training data, based on the score ratio.
    The number of examples per post is limited to prevent over-fitting.
    """
    def __init__(self, ratio_thresh: float, examples_per_post: int):
        self.ratio_thresh = ratio_thresh
        self.examples_per_post = examples_per_post

    def subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["score_ratio"] >= self.ratio_thresh]
        df = df.groupby("post_id").apply(
            lambda x: x.sample(n=min(self.examples_per_post, len(x)))
        )
        df = df.sample(n=len(df))
        return df


def clean_text(text: str) -> str:
    return text.replace("\n", " ")


def format_t5(
    name: str,
    out_dir: str = "data",
    domains: List[str] = SUBREDDITS,
    max_tokens: int = 500,
    subsampler: SubSampler = None,
):
    """
    Format the training data across all domains specified and write
    to a file at {out_dir}/{name}_{train/test/validation}.json.

    If the input doesn't naturally fit into max_tokens tokens, then
    use pybsd to truncate the context such that it fits under the limit.
    If the responses alone are greater than max_tokens tokens,
    ignore the example (for all the splits of data).

    If a SubSampler is specified, use it to subsample ONLY the
    training data (the test data should be left untouched).
    """
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    segmenter = pysbd.Segmenter(language="en", clean=False)

    for split in ["train", "validation", "test"]:
        f = open(f"{out_dir}/{name}_{split}.json", "w")

        for domain in domains:
            print(domain, split)

            if domain in SUBREDDITS:
                df = load_dataset("stanfordnlp/shp", data_dir=domain)[split].to_pandas()
            elif domain in ANTHROPIC:
                if split == "validation": continue
                df = pd.read_json(f"data/{domain}/{split}.json", orient="records", lines=True)
            else:
                raise Exception

            # subsampling (only subsample training data)
            if subsampler is not None and split == "train" and domain in SUBREDDITS:
                df = subsampler.subsample(df)

            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                prompt = ""
                prompt = (
                    prompt
                    + "\n\n RESPONSE A: "
                    + clean_text(row["human_ref_A"])
                    + "\n\n RESPONSE B: "
                    + clean_text(row["human_ref_B"])
                )
                prompt = prompt + "\n\n Which response is better? RESPONSE"
                target = "A" if row["labels"] == 1 else "B"

                slack = max_tokens - len(tokenizer(prompt).input_ids)
                sentences = []

                if domain in SUBREDDITS:
                    for s in segmenter.segment(clean_text(row["history"])):
                        slack -= len(tokenizer(s).input_ids)

                        if slack > 0:
                            sentences.append(s)

                    prompt = "POST: " + "".join(sentences) + " " + prompt
                else:
                    prompt = "POST: " + row["history"] + " " + prompt

                if len(tokenizer(prompt).input_ids) > max_tokens:
                    continue

                if domain in ANTHROPIC:
                    f.write(
                        json.dumps(
                            {
                                "x": prompt,
                                "y": target,
                                "domain": "anthropic",
                                "post_id": "",
                                "post_upvote_ratio": 1.0,
                                "score_ratio": 1,
                            }
                        )
                    )
                else:
                    f.write(
                        json.dumps(
                            {
                                "x": prompt,
                                "y": target,
                                "domain": row["domain"],
                                "post_id": row["post_id"],
                                "post_upvote_ratio": row["upvote_ratio"],
                                "score_ratio": row["score_ratio"],
                            }
                        )
                    )

                f.write("\n")

        f.close()


if __name__ == "__main__":
    format_t5(
        "combined",
        domains=(SUBREDDITS + ANTHROPIC),
        max_tokens=500,
        subsampler=RatioSampler(2.0, 5),
    )
