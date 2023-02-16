import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
from datetime import date
from pathlib import Path

import click
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, T5ForConditionalGeneration, MT5ForConditionalGeneration,BartForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer
from transformers import BartTokenizerFast as BartTokenizer

import wandb
from utils import *

today = date.today()


@click.command()
@click.option(
    "-ms",
    "--model_size",
    default="small",
    show_default=True,
    type=str,
    help="Select model: small, base, large",
)
@click.option(
    "-n",
    "--name",
    default="t5-",
    show_default=True,
    type=str,
    help="Select model: t5-, google/mt5-, facebook/bart-",
)
@click.option(
    "-d",
    "--data_dir",
    default="webNLG2020_prepared_data/",
    show_default=True,
    type=str,
    help="Directory with files named 'train.csv', 'validation.csv', 'test.csv'. For use on WebNLG data set run src/prepare_data first",
)
@click.option(
    "-s",
    "--data_set",
    default="WebNLG",
    show_default=True,
    type=str,
    help="Select of the data set: 'WebNLG', 'AXgen', 'AXgen'",
)
@click.option(
    "-in",
    "--input_len",
    type=int,
    help="Maximum_input_length -- if not used script will take longer",
)
@click.option(
    "-out",
    "--output_len",
    type=int,
    help="Maximum_output_length -- if not used script will take longer",
)
@click.option(
    "-t",
    "--do_test",
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help="Test the model on the test set after training",
)
@click.option(
    "-md",
    "--model_directory",
    show_default=True,
    default="models",
    type=str,
    help="Directory to save the model",
)
@click.option(
    "-gpu",
    "--gpu",
    show_default=True,
    default=0,
    type=int,
    help="Select which GPU to use if there are multiple",
)
@click.option(
    "-e",
    "--n_epochs",
    show_default=True,
    default=15,
    type=int,
    help="Number of epochs -- Recommended value for t5-small: 15 -- Recommended value for t5-base and t5-large: 8",
)
@click.option(
    "-b",
    "--batch_size",
    show_default=True,
    default=20,
    type=int,
    help="Batch size -- Recommended value for t5-small: 20 -- Recommended value for t5-base and t5-large: 4",
)
@click.option(
    "-lr",
    "--learning_rate",
    show_default=True,
    default=1e-05,
    type=float,
    help="Learning rate -- Recommended value for t5-small: 1e-05 -- Recommended value for t5-base and t5-large: 5e-06",
)
@click.option(
    "-opt",
    "--optimizer",
    show_default=True,
    default="AdamW",
    type=str,
    help="Optimizer -- Recommended value for t5-small: 'AdamW' -- Recommended value for t5-base and t5-large: 'Adafactor'",
)
@click.option(
    "-dr",
    "--dropout",
    show_default=True,
    default=0.1,
    type=int,
    help="dropout rate",
)
def train(
    model_size: str,
    name: str,
    data_dir: str,
    data_set: str,
    input_len: int,
    output_len: int,
    do_test: bool,
    model_directory: str,
    gpu: int,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    dropout: int,
):
    pl.seed_everything(42)
    model_name = name + model_size
    if name == "t5-":
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif name == "google/mt5-":
        model = MT5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif name == "facebook/bart-":
        model = BartForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
    MODEL_PATH = f"good_models/{today}_{model_name.split('/')[-1]}_{data_set}_e{n_epochs}_b{batch_size}_{optimizer}_{learning_rate}"
    RUN_NAME = MODEL_PATH.split("/")[-1]

    train_df = pd.read_csv(data_dir + "train.csv", names=["data_in", "text"], sep="\t")
    dev_df = pd.read_csv(
        data_dir + "validation.csv", names=["data_in", "text"], sep="\t"
    )
    test_df = pd.read_csv(data_dir + "test.csv", names=["data_in", "text"], sep="\t")

    if not input_len and not output_len:
        input_len, output_len = get_max_lengths(train_df, dev_df, test_df, tokenizer)

    data_module = SenGenDataModule(
        train_df, test_df, dev_df, tokenizer, batch_size, input_len, output_len
    )

    sen_gen = SenGenModel(
        model, tokenizer, input_len, output_len, optimizer, learning_rate
    )
    sen_gen.model.config.dropout_rate=dropout
    logger = WandbLogger(name=RUN_NAME, save_dir=model_directory, project=data_set)
    checkpoint_callback = ModelCheckpoint(
        verbose=True, dirpath=model_directory, filename=RUN_NAME, save_weights_only=True
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=n_epochs,
        gpus=[gpu],
        progress_bar_refresh_rate=10,
    )

    trainer.fit(sen_gen, data_module)

    if do_test:
        trainer.test(sen_gen, datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    train()
