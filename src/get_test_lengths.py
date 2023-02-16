import click
import pandas as pd
import pytorch_lightning as pl
import torch
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
    "-d",
    "--data_dir",
    type=str,
    help="Path to the data directory",
)
@click.option(
    "-m1",
    "--model1_name",
    type=str,
    help="Choose: t5-, google/mt5-, facebook/bart-",
)
@click.option(
    "-m2",
    "--model2_name",
    type=str,
    help="Choose: t5-, google/mt5-, facebook/bart-",
)
@click.option(
    "-f",
    "--first",
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help="Test the model on the test set after training",
)
def get_test_lengths(data_dir, model1_name, model2_name, first):
    if first:
        size = "large"
    else:
        size = "small"
    for model_name in [model1_name, model2_name]:
        if "t5" in model_name:
            tokenizer = T5Tokenizer.from_pretrained(model_name + size)
        else:
            tokenizer = BartTokenizer.from_pretrained(model_name + size)

        test_dfs = list()
        for i in range(5):
            test_df = pd.read_table(data_dir + f"references/test{i}", names = [i])
            test_dfs.append(test_df)
        test_dfs = pd.concat(test_dfs, axis=1)
        
        text_lens = list()
        for j, row in test_dfs.iterrows():
            min_text_len = None
            for i in range(5):
                if str(row[i]) != "nan":
                    text_encoding = tokenizer(row[i])
                    if not min_text_len:
                        min_text_len = len(text_encoding.tokens())
                    elif len(text_encoding.tokens()) <= min_text_len:
                        min_text_len = len(text_encoding.tokens())
            text_lens.append(min_text_len)
            
        assert len(test_dfs) == len(text_lens)
        assert None not in text_lens
        with open(data_dir + "test_lens_" + model_name.split("/")[-1][:-1], "w") as f:
            for i in range(len(text_lens)):
                if i != len(text_lens) - 1:
                    f.write(str(text_lens[i]) + "\n")
                else:
                    f.write(str(text_lens[i]))


if __name__ == "__main__":
    get_test_lengths()
