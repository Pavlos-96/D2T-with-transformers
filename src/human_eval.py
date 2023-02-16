import glob
import os
import re
from pathlib import Path

import click
import pandas as pd
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pprint
import statsmodels
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from statsmodels.stats.descriptivestats import sign_test#, ttest_1samp


def main(): 
    np.random.seed(6)
    df = pd.read_csv("human_eval/uncleaned_results.csv", index_col="Unnamed: 0").drop("language level", axis=1)
    df = df.T.sort_values("dataset")
    aggr_df, _ = aggregate_raters(df.T[:22].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("kappa before cleaning = ", fleiss)
    df = pd.read_csv("human_eval/cleaned_results.csv", index_col="Unnamed: 0").drop("language level", axis=1)
    df = df.T.sort_values("dataset")
    aggr_df, _ = aggregate_raters(df.T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("kappa after cleaning = ", fleiss)
    df_systems = df.sort_values("name")
    aggr_df, _ = aggregate_raters(df_systems[:12].T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("mt5vstrue = ", fleiss)
    aggr_df, _ = aggregate_raters(df_systems[12:24].T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("t5vsmt5 = ", fleiss)
    aggr_df, _ = aggregate_raters(df_systems[24:].T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("truevst5 = ", fleiss)
    
    df_questions = df.sort_values("question")
    aggr_df, _ = aggregate_raters(df_questions[:18].T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("structure = ", fleiss)
    aggr_df, _ = aggregate_raters(df_questions[18:].T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("fluency = ", fleiss)
    
    df_ax = df[:int(len(df)/2)].sort_values("question")
    aggr_df, _ = aggregate_raters(df_ax.T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("kappa ax = ", fleiss)
    df_ax_structure = df_ax[:int(len(df_ax)/2)].sort_values("name").T
    df_ax_fluency = df_ax[int(len(df_ax)/2):].sort_values("name").T
    df_webnlg = df[int(len(df)/2):].sort_values("question")
    aggr_df, _ = aggregate_raters(df_webnlg.T[:18].T.astype(int).to_numpy())
    fleiss = fleiss_kappa(aggr_df, method='fleiss')
    print("kappa webNLG = ", fleiss)
    df_webnlg_structure = df_webnlg[:int(len(df_webnlg)/2)].sort_values("name").T
    df_webnlg_fluency = df_webnlg[int(len(df_webnlg)/2):].sort_values("name").T
    names = ["ax_structure_", "ax_fluency_", "webnlg_structure_", "webnlg_fluency_"]
    comparisons = ["mt5vstrue", "t5vsmt5", "truevst5"]
    data = dict()
    reg_list = list()
    sign_test_results = dict()
    kappa_systems = {"mt5vstrue": [], "t5vsmt5": [], "truevst5": []}
    for i, d in enumerate([df_ax_structure, df_ax_fluency, df_webnlg_structure, df_webnlg_fluency]):
        a = d[:18].T[:int(len(d.columns)/3)].T.to_numpy().flatten().astype(int)
        kappa_systems[comparisons[0]].append(a)
        data[names[i] + comparisons[0]] = f"{str(len(a[a == 0]))} & {str(len(a[a == 1]))} & {str(len(a[a == 2]))}"
        a = d[:18].T[int(len(d.columns)/3):2*int(len(d.columns)/3)].T.to_numpy().flatten().astype(int)
        kappa_systems[comparisons[1]].append(a)
        data[names[i] + comparisons[1]] = f"{str(len(a[a == 0]))} & {str(len(a[a == 1]))} & {str(len(a[a == 2]))}"
        a = d[:18].T[2*int(len(d.columns)/3):].T.to_numpy().flatten().astype(int)
        kappa_systems[comparisons[2]].append(a)
        data[names[i] + comparisons[2]] = f"{str(len(a[a == 0]))} & {str(len(a[a == 1]))} & {str(len(a[a == 2]))}"
        b = d[:18].T[:int(len(d.columns)/3)].T.to_numpy().flatten().astype(int)
        b[b == 1] = -1
        b[b == 2] = 1
        sign_test_results[names[i] + comparisons[0]] = (b.mean(), sign_test(samp=b,mu0=0))
        b = d[:18].T[int(len(d.columns)/3):2*int(len(d.columns)/3)].T.to_numpy().flatten().astype(int)
        b[b == 1] = -1
        b[b == 2] = 1
        sign_test_results[names[i] + comparisons[1]] = (b.mean(), sign_test(samp=b,mu0=0))
        b = d[:18].T[2*int(len(d.columns)/3):].T.to_numpy().flatten().astype(int)
        b[b == 1] = -1
        b[b == 2] = 1
        sign_test_results[names[i] + comparisons[2]] = (b.mean(), sign_test(samp=b,mu0=0))
        aggr_df, _ = aggregate_raters(d[:18].T.astype(int).to_numpy())
        fleiss = fleiss_kappa(aggr_df, method='fleiss')
        print(f"kappa {names[i]} = ", fleiss)
    for k, v in kappa_systems.items():
        kappa_systems[k] = np.concatenate(tuple(v))
        fleiss = fleiss_kappa(aggr_df, method='fleiss')
    #pprint.pprint(kappa_systems)
    pprint.pprint(data)
    pprint.pprint(sign_test_results)
    
if __name__ == "__main__":
    main()
    