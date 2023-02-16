import glob
import os
import re
from pathlib import Path

import click
import pandas as pd
import pytorch_lightning as pl
import torch
import math
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AdamW, T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer
from transformers import BartTokenizerFast as BartTokenizer

from utils import *


@click.command()
@click.option(
    "-m",
    "--model_path",
    type=str,
    help="Path to the model",
)
@click.option(
    "-d",
    "--data_file",
    type=str,
    help="Path to the data file to predict on",
)
@click.option(
    "-gpu",
    "--gpu",
    show_default=True,
    default=0,
    type=int,
    help="Select which GPU to use if there are multiple",
)
def predict(model_path, data_file, gpu):
    short_name = model_path.split("/")[-1].split("_")[1]
    if short_name.startswith("mt5"):
        model_name = "google/" + short_name
    elif short_name.startswith("bart"):
        model_name = "facebook/" + short_name
    elif short_name.startswith("t5"):
        model_name = short_name
    if "t5-" in model_path:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif "bart-" in model_path:
        tokenizer = BartTokenizer.from_pretrained(model_name)
    RUN_NAME = model_path.split("/")[-1]
    trained_model = SenGenModel.load_from_checkpoint(model_path)

    trained_model.freeze()
    if torch.cuda.is_available():
        trained_model.to(torch.device(f"cuda:{gpu}"))

    if data_file:
        to_predict = pd.read_csv(data_file, names=["rdf"], sep="\t")
        if not os.path.exists(f"predictions/{RUN_NAME[:-5]}_preds.txt"):
            with open(f"predictions/{RUN_NAME[:-5]}_preds.txt", "w") as f:
                for i in range(len(to_predict)):
                    generated_text = trained_model.generate_sentence(
                        to_predict.iloc[i][0], gpu=gpu, do_sample=False
                    )
                    if i != len(to_predict) - 1:
                        f.write(generated_text + "\n")
                    else:
                        f.write(generated_text)

def predict_df(model_path, df, gpu, min_length=None):
    short_name = model_path.split("/")[-1].split("_")[1]
    if short_name.startswith("mt5"):
        model_name = "google/" + short_name
    elif short_name.startswith("bart"):
        model_name = "facebook/" + short_name
    elif short_name.startswith("t5"):
        model_name = short_name
    if "t5-" in model_path:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif "bart-" in model_path:
        tokenizer = BartTokenizer.from_pretrained(model_name)
    RUN_NAME = model_path.split("/")[-1]
    trained_model = SenGenModel.load_from_checkpoint(model_path)

    trained_model.freeze()
    if torch.cuda.is_available():
        trained_model.to(torch.device(f"cuda:{gpu}"))
    texts = list()
    for i in range(len(df)):
        if len(df.columns) == 2:
            generated_text = trained_model.generate_sentence(df["to_predict"].iloc[i], gpu=gpu, min_length=int(round(0.8 * int(df[df.columns[-1]].iloc[i]))), do_sample=False)
        else:
            if min_length:
                generated_text = trained_model.generate_sentence(df["to_predict"].iloc[i], gpu=gpu, do_sample=False, min_length=int(math.log(len(df["to_predict"].iloc[i].split(";")), 5) * 50))
            else:
                generated_text = trained_model.generate_sentence(df["to_predict"].iloc[i], gpu=gpu, do_sample=False)
        texts.append(generated_text)
    return pd.DataFrame(texts)

if __name__ == "__main__":
    predict()
