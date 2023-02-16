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
            if categories[i] == "Weiße Ware" or "Mobiltelefone":
                m = 2
            else:
                m = 1
            category_df = df[df['to_predict'].str.contains(categories[i])]
            category_df = category_df.sample(frac=1, random_state=0)
            dfs.append(category_df[:int(m * 3)])
            df.drop(list(category_df[:int(m * 3)].index))
        df = pd.concat(dfs, axis=0)
    else:
        df = df.sample(frac=1, random_state=1)
        df = df[:int(n)]
    preds1 = predict_df(model1_path, df[["to_predict", "lens1"]], gpu) #t5
    preds2 = predict_df(model2_path, df[["to_predict", "lens2"]], gpu) #mt5
    df_true = pd.DataFrame(df[["true"]].values)
    df_index = pd.DataFrame(df[["index"]].values)
    df = pd.concat([preds1, preds2, df_true, df_index], axis=1)
    df.columns = ["t5", "mt5", "true", "index"]
    comparisons = [["t5", "mt5"], ["mt5", "true"], ["true", "t5"]]
    new_dict = {"index": [], "dataset": [], "question": [], "name": [], "text1": [], "text2": []}
    i = 0
    for question in ["grammatisch", "lesbar"]:
        for h in range(int(n/2/3)):
            for j in range(len(comparisons)):
                if ax:
                    new_dict["dataset"].append("ax")
                else:
                    new_dict["dataset"].append("webnlg")
                new_dict["question"].append(question)
                new_dict["name"].append("vs".join(comparisons[j]))
                new_dict["text1"].append(df[comparisons[j][0]].iloc[i].replace("➤", ""))
                new_dict["text2"].append(df[comparisons[j][1]].iloc[i].replace("➤", ""))
                new_dict["index"].append(df["index"].iloc[i])
                i += 1
    df = pd.DataFrame(new_dict)
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
    
    
    num = 36
    df1 = generate_sentences(model1_path, model2_path, ax_data_dir, int(num/2), gpu, ax=True)
    df2 = generate_sentences(model3_path, model4_path, webnlg_data_dir, int(num/2), gpu, ax=False)
    df = pd.concat([df1, df2], axis=0)
    df.sort_values("question", inplace=True)
    df = pd.concat([df[:int(len(df)/2)].sample(frac=1, random_state=1), df[int(len(df)/2):].sample(frac=1, random_state=1)])
    df.to_csv("second_survey_data.csv", sep="\t", index=None)

    
if __name__ == "__main__":
    prepare_survey()
    