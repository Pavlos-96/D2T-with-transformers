import glob
import os
import re
from pathlib import Path

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

from predict import predict_df
import random

def generate_sentences(model1_path, model2_path, data_dir, n, gpu):
    to_predict = pd.read_table(data_dir + "to_predict", header=None, names=["to_predict"])
    true = pd.read_table(data_dir + "references/test0", header=None, names=["true"])
    cats = pd.read_csv(data_dir + "test_cats.csv")
    df_index = pd.DataFrame(range(len(true)), columns=["index"])
    df = pd.concat([cats, to_predict, true, df_index], axis=1)
    
    df_unseen = df[df["categories"].isin(["Film", "Scientist", "MusicalWork"])]
    df = df.drop(index=df_unseen.index)

    df = df.sample(frac=1, random_state=1)
    df_unseen = df_unseen.sample(frac=1, random_state=2)
    
    df = pd.concat([df[:int(n/2)], df_unseen[:int(n/2)]], axis=0)
    df = df.sample(frac=1, random_state=1)
    print(df)
    preds1 = predict_df(model1_path, df[["to_predict"]], gpu)
    preds2 = predict_df(model2_path, df[["to_predict"]], gpu)
    df_index = pd.DataFrame(df[["index"]].values)
    input_df = pd.DataFrame(df[["to_predict"]].values)
    cats_df = pd.DataFrame(df[["categories"]].values)
    true = pd.DataFrame(df[["true"]].values)
    df = pd.concat([true, cats_df, input_df, preds1, preds2, df_index], axis=1)
    df.columns = ["true","categories", "input", "BART", "T5", "index"]
    comparisons = [["BART", "T5"], ["T5", "BART"]]
    new_dict = {"true": [], "categories": [], "input": [], "index": [], "name": [], "text1": [], "text2": []}
    i = 0
    for j in range(len(comparisons)):
        for h in range(int(n/2)):
            new_dict["name"].append("vs".join(comparisons[j]))
            new_dict["text1"].append(df[comparisons[j][0]].iloc[i])
            new_dict["text2"].append(df[comparisons[j][1]].iloc[i])
            new_dict["index"].append(df["index"].iloc[i])
            new_dict["input"].append(df["input"].iloc[i])
            new_dict["categories"].append(df["categories"].iloc[i])
            new_dict["true"].append(df["true"].iloc[i])
            i += 1
    df = pd.DataFrame(new_dict).sample(frac=1, random_state=1)
    return df

@click.command()
@click.option(
    "-m1",
    "--model1_path",
    type=str,
    help="BART",
)
@click.option(
    "-m2",
    "--model2_path",
    type=str,
    help="T5",
)
@click.option(
    "-wd",
    "--webnlg_data_dir",
    type=str,
    help="",
)
@click.option(
    "-gpu",
    "--gpu",
    show_default=True,
    default=0,
    type=int,
    help="Select which GPU to use if there are multiple",
)
def prepare_survey(model1_path, model2_path, webnlg_data_dir, gpu): 
    num = 20
    df = generate_sentences(model1_path, model2_path, webnlg_data_dir, int(num), gpu)
    
    df[["index", "categories", "name"]].to_csv("first_survey_hidden.csv", sep="\t", index=None)
    
    df[["Data Coverage", "Relevance", "Correctness", "Text Structure", "Fluency"]] = ""
    df[["input", "true", "text1", "text2", "Data Coverage", "Relevance", "Correctness", "Text Structure", "Fluency"]].to_csv("first_survey_data.csv", sep="\t", index=None)

    
if __name__ == "__main__":
    prepare_survey()
    