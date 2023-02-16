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

from predict import predict_df
import random

def generate_sentences(model1_path, model2_path, data_dir, n, gpu, ax):
    to_predict = pd.read_table(data_dir + "to_predict", header=None, names=["to_predict"])
    lens1 = pd.read_table(data_dir + "test_lens_" + model1_path.split("/")[-1].split("_")[1].split("-")[0], header=None, names=["lens1"])
    lens2 = pd.read_table(data_dir + "test_lens_" + model2_path.split("/")[-1].split("_")[1].split("-")[0], header=None, names=["lens2"])
    true = pd.read_table(data_dir + "true", header=None, names=["true"])
    df_index = pd.DataFrame(range(len(true)), columns=["index"])
    df = pd.concat([to_predict, lens1, lens2, true, df_index], axis=1)
    if ax:
        df = df[df['to_predict'].str.contains("de_DE")]
        categories = ["Mode", "Gartenmöbel", "Weiße Ware", "Mobiltelefone"]
        dfs = list()
        for i in range(len(categories)):
            # if categories[i] in ["Weiße Ware", "Gartenmöbel"]:
            #     m = 3
            # else:
            m = 2
            category_df = df[df['to_predict'].str.contains(categories[i])]
            category_df = category_df.sample(frac=1, random_state=32)
            dfs.append(category_df[:int(m)])
            df.drop(list(category_df[:int(m)].index))
        df = pd.concat(dfs, axis=0)
    else:
        df = df.sample(frac=1, random_state=32)
        df = df[:int(n/2)]
    preds1 = predict_df(model1_path, df[["to_predict", "lens1"]], gpu)
    preds2 = predict_df(model2_path, df[["to_predict", "lens2"]], gpu)
    df_index = pd.DataFrame(df[["index"]].values)
    df_to_predict = pd.DataFrame(df[["to_predict"]].values)
    df = pd.concat([df_to_predict, preds1, preds2, df_index], axis=1)
    df.columns = ["input", "t5", "mt5", "index"]

    hand = pd.read_csv("second_manual_eval/handwritten_for_man_2.csv")
    hand.columns = ["to_predict"]
    if ax:
        min_length = "hand"
        hand = hand[:8]
    else:
        min_length = None
        hand = hand[8:]
    hand_preds1 = predict_df(model1_path, hand[["to_predict"]], gpu, min_length)
    hand_preds2 = predict_df(model2_path, hand[["to_predict"]], gpu, min_length)
    hand_df_input = pd.DataFrame(hand[["to_predict"]].values)
    hand_df = pd.concat([hand_df_input, hand_preds1, hand_preds2], axis=1)
    hand_df[len(hand_df.columns)] = "n"

    hand_df.columns = ["input", "t5", "mt5", "index"]
    df = pd.concat([df, hand_df], axis=0)
    comparisons = [["t5", "mt5"], ["mt5", "t5"]]
    new_dict = {"input": [], "index": [], "dataset": [], "name": [], "text1": [], "text2": []}
    i = 0
    for h in range(int(n/2)):
        for j in range(len(comparisons)):
            if ax:
                new_dict["dataset"].append("ax")
            else:
                new_dict["dataset"].append("webnlg")
            new_dict["name"].append("vs".join(comparisons[j]))
            if ax:
                new_dict["input"].append(df["input"].iloc[i].replace("; ", "\n "))
            else:
                new_dict["input"].append(df["input"].iloc[i].replace("&&", "\n"))
            new_dict["text1"].append(str(df[comparisons[j][0]].iloc[i]).replace("➤", ""))
            new_dict["text2"].append(str(df[comparisons[j][1]].iloc[i]).replace("➤", ""))
            new_dict["index"].append(df["index"].iloc[i])
            i += 1
    df = pd.DataFrame(new_dict)
    print(len(df))
    return df

@click.command()
@click.option(
    "-m1",
    "--model1_path",
    type=str,
    help="t5 for ax dataset",
)
@click.option(
    "-m2",
    "--model2_path",
    type=str,
    help="mt5 for ax dataset",
)
@click.option(
    "-m3",
    "--model3_path",
    type=str,
    help="t5 for german weblg dataset",
)
@click.option(
    "-m4",
    "--model4_path",
    type=str,
    help="mt5 for german weblg dataset",
)
@click.option(
    "-ad",
    "--ax_data_dir",
    type=str,
    help="",
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
def prepare_survey(model1_path, model2_path, model3_path, model4_path, ax_data_dir, webnlg_data_dir, gpu):
    assert model1_path.split("/")[-1].split("_")[1].split("-")[0] == model3_path.split("/")[-1].split("_")[1].split("-")[0] == "t5" and  model2_path.split("/")[-1].split("_")[1].split("-")[0] == model4_path.split("/") [-1].split("_")[1].split("-")[0] == "mt5"
    
    
    num = 16
    df1 = generate_sentences(model1_path, model2_path, ax_data_dir, int(num), gpu, ax=True)
    df2 = generate_sentences(model3_path, model4_path, webnlg_data_dir, int(num), gpu, ax=False)
    print(len(df1), len(df2))
    df = pd.concat([df1, df2], axis=0)
    df = df.sample(frac=1, random_state=1)
    print(len(df))
    df[["index", "name", "dataset"]].to_csv("second_manual_eval_hidden.csv", sep="\t", index=None)
    
    
    df[["Data Coverage", "Relevance", "Correctness", "Text Structure", "Fluency"]] = ""
    df[["input", "text1", "text2", "Data Coverage", "Relevance", "Correctness", "Text Structure", "Fluency"]].to_csv("second_manual_eval.csv", sep="\t", index=None)
    print(len(df))

    
if __name__ == "__main__":
    prepare_survey()
    