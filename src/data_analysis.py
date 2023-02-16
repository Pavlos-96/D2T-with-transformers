import glob
import os
import re
from pathlib import Path

import click
import pandas as pd
import random
import nltk
import math
from nltk.tokenize import word_tokenize, sent_tokenize
import statistics


def read_data(datadir_path):
    train = pd.read_csv(datadir_path + "train.csv", sep="\t", names=["data_in", "data_out"])
    dev = pd.read_csv(datadir_path + "validation.csv", sep="\t", names=["data_in", "data_out"])
    test = pd.read_csv(datadir_path + "test.csv", sep="\t", names=["data_in", "data_out"])
    return pd.concat([train, dev, test])

def get_num_attr(df, data):
    attributes = set()
    if data == "webnlg":
        for index, row in df.iterrows():
            for triple in row["data_in"].split(" && "):
                attributes.add(triple.split(" | ")[1])
    else:
        for index, row in df.iterrows():
            for att in row["data_in"].split("; "):
                attributes.add(att.split("➤")[0])
    return len(attributes)

def get_num_input_pattern(df, data):
    input_patterns = list()
    if data == "webnlg":
        for index, row in df.iterrows():
            input_patterns.append(frozenset(triple.split(" | ")[1] for triple in row["data_in"].split(" && ")))
    else:
        for index, row in df.iterrows():
            input_patterns.append(frozenset(att.split("➤")[0] for att in row["data_in"].split("; ")))
    return len(set(input_patterns))


def get_num_input(df):
    return len(df.drop_duplicates(subset="data_in"))

def get_num_data_text(df):
    return len(df["data_in"])

def get_text_len(df):
    lens = list()
    for index, row in df.iterrows():
        lens.append(len(word_tokenize(row["data_out"].replace("➤", ""))))
    return sum(lens) / len(lens), statistics.median(lens), min(lens), max(lens)

def get_sent_len(df):
    lens = list()
    for index, row in df.iterrows():
        lens.append(len(sent_tokenize(row["data_out"].replace("➤", ""))))
    return sum(lens) / len(lens), statistics.median(lens), min(lens), max(lens)

def types_and_tokens(df):
    voc = dict()
    for index, row in df.iterrows():
        text = word_tokenize(row["data_out"].replace("➤", ""))
        for word in text:
            voc.setdefault(word, 0)
            voc[word] += 1
    tokens = 0
    for k, v in voc.items():
        tokens += v
    cttr = len(voc) / math.sqrt(2*tokens)
    return len(voc), tokens, cttr



@click.command()
@click.option(
    "-w",
    "--webnlg",
    type=str,
    help="path to webnlg data directory",
)
@click.option(
    "-wo",
    "--webnlg_original",
    type=str,
    help="path to webnlg_original data directory",
)
@click.option(
    "-a",
    "--ax",
    type=str,
    help="path to ax data directory",
)
def main(webnlg_original, webnlg, ax): 
    webnlg_original = read_data(webnlg_original)
    webnlg = read_data(webnlg)
    ax = read_data(ax)
#     analysis = {"ax": {"Nb. Input": get_num_input(ax), "Nb. Data-Text Pairs": get_num_data_text(ax), "Nb. Domains": 4, "Nb. Attributes": get_num_attr(ax, "ax"), "Nb. Input Patterns": get_num_input_pattern(ax, "ax"), "Nb. Input / Nb Input Pattern": get_num_input(ax)/get_num_input_pattern(ax, "ax")},
#                "webnlg": {"Nb. Input": get_num_input(webnlg), "Nb. Data-Text Pairs": get_num_data_text(webnlg), "Nb. Domains": 10, "Nb. Attributes": get_num_attr(webnlg, "webnlg"), "Nb. Input Patterns": get_num_input_pattern(webnlg, "webnlg"), "Nb. Input / Nb Input Pattern": get_num_input(webnlg)/get_num_input_pattern(webnlg, "webnlg")},
#                "webnlg_original": {"Nb. Input": get_num_input(webnlg_original), "Nb. Data-Text Pairs": get_num_data_text(webnlg_original), "Nb. Domains": 19, "Nb. Attributes": get_num_attr(webnlg_original, "webnlg"), "Nb. Input Patterns": get_num_input_pattern(webnlg_original, "webnlg"), "Nb. Input / Nb Input Pattern": get_num_input(webnlg_original)/get_num_input_pattern(webnlg_original, "webnlg")}}
#     analysis_df = pd.DataFrame(analysis)
#     print(analysis_df)
    
    # average, median, min, max = get_text_len(ax)
    # print("AX: ", average, median, min, max)
    # average, median, min, max = get_text_len(webnlg)
    # print("WebNLG_ger: ", average, median, min, max)
    # average, median, min, max = get_text_len(webnlg_original)
    # print("WebNLG_original: ", average, median, min, max)
    
    # average, median, min, max = get_sent_len(ax)
    # print("AX: ", average, median, min, max)
    # average, median, min, max = get_sent_len(webnlg)
    # print("WebNLG_ger: ", average, median, min, max)
    # average, median, min, max = get_sent_len(webnlg_original)
    # print("WebNLG_original: ", average, median, min, max)
    
    types, tokens, cttr = types_and_tokens(ax)
    print("AX: ", types, tokens, cttr)
    types, tokens, cttr = types_and_tokens(webnlg)
    print("WebNLG_ger: ", types, tokens, cttr)
    types, tokens, cttr = types_and_tokens(webnlg_original)
    print("WebNLG_original: ", types, tokens, cttr)

if __name__ == "__main__":
    main()
