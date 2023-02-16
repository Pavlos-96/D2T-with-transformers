import glob
import os
import re
from pathlib import Path

import click
import pandas as pd
import random


def make_eval(df, comp_model, second_eval=None):
    stats = {
     'Vote': [f'{comp_model}', 'T5', 'None'],
     'Data Coverage': [0, 0, 0],
     'Relevance': [0, 0, 0],
     'Correctness': [0, 0, 0],
     'Text Structure': [0, 0, 0],
     'Fluency': [0, 0, 0]}
    stats_df = pd.DataFrame(data=stats)
    stats_df.set_index("Vote", inplace=True)

    for index, row in df.iterrows():
        for col in stats_df.columns:
            choice = ["None"] + [row["name"].split("vs")[i].upper() for i in range(len(row["name"].split("vs")))]
            stats_df[col].loc[choice[int(row[col])]] += 1
    print("overall:\n", stats_df, "\n\n")
    
    stats = {
     'Vote': [f'{comp_model}', 'T5', 'None'],
     'Data Coverage': [0, 0, 0],
     'Relevance': [0, 0, 0],
     'Correctness': [0, 0, 0],
     'Text Structure': [0, 0, 0],
     'Fluency': [0, 0, 0]}
    stats_df_known = pd.DataFrame(data=stats)
    stats_df_known.set_index("Vote", inplace=True)

    for index, row in df.iterrows():
        for col in stats_df_known.columns:
            if (not second_eval and row["categories"] not in ["Film", "Scientist", "MusicalWork"]) or (second_eval and row["index"] != "n"):
                choice = ["None"] + [row["name"].split("vs")[i].upper() for i in range(len(row["name"].split("vs")))]
                stats_df_known[col].loc[choice[int(row[col])]] += 1
    print("known:\n", stats_df_known, "\n\n")
    
    stats = {
     'Vote': [f'{comp_model}', 'T5', 'None'],
     'Data Coverage': [0, 0, 0],
     'Relevance': [0, 0, 0],
     'Correctness': [0, 0, 0],
     'Text Structure': [0, 0, 0],
     'Fluency': [0, 0, 0]}
    stats_df_unknown = pd.DataFrame(data=stats)
    stats_df_unknown.set_index("Vote", inplace=True)

    for index, row in df.iterrows():
        for col in stats_df_unknown.columns:
            if (not second_eval and row["categories"] in ["Film", "Scientist", "MusicalWork"]) or (second_eval and row["index"] == "n"):
                choice = ["None"] + [row["name"].split("vs")[i].upper() for i in range(len(row["name"].split("vs")))]
                stats_df_unknown[col].loc[choice[int(row[col])]] += 1
    print("unknown:\n", stats_df_unknown)

@click.command()
@click.option(
    "-f",
    "--filled",
    type=str,
    help="path to filled survey",
)
@click.option(
    "-h",
    "--hidden",
    type=str,
    help="path to hidden info",
)
@click.option(
    "-s",
    "--second_eval",
    is_flag=True,
    show_default=True,
    default=None,
    type=bool,
    help="Test the model on the test set after training",
)
def prepare_survey(filled, hidden, second_eval): 
    filled = pd.read_csv(filled, sep=";")
    hidden = pd.read_csv(hidden, sep="\t")
    df = pd.concat([hidden, filled], axis=1)
    if second_eval:
        print("\nwebnlg\n")
        make_eval(df[df["dataset"] == "webnlg"], "MT5", second_eval)
        print("\n\nax\n")
        make_eval(df[df["dataset"] == "ax"], "MT5", second_eval)
    else: 
        make_eval(df, "BART")

    

if __name__ == "__main__":
    prepare_survey()
    