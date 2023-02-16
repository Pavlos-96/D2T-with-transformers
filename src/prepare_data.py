import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd
import click

@click.command()
@click.option(
    "-d",
    "--dataset",
    default="english",
    show_default=True,
    type=str,
    help="choose: webnlg_english, webnlg_german, ax",
)
@click.option(
    "-p",
    "--path",
    type=str,
    help="",
)
def prepare_data(dataset, path):
    train_path = "train/**/*"
    dev_path = "dev/**/*"

    if dataset == "webnlg_english":
        test_path = "test/rdf-to-text-generation-test-data-with-refs-en"
        sets = [train_path, dev_path, test_path]
        Path("webNLG2020_prepared_data").mkdir(parents=True, exist_ok=True)
        for s in sets:
            files = glob.glob(
                path + "release_v3.0/en/" + s + ".xml", recursive=True
            )
            test_data = dict()
            data = {"rdf": [], "text": []}
            cats = {"domain": []}
            for file in files:
                tree = ET.parse(file)
                root = tree.getroot()
                for sub_root in root:
                    for entry in sub_root:
                        for element in entry:
                            if "modifiedtripleset" in str(element):
                                inp = " && ".join([triple.text for triple in element])
                            if "lex" in str(element):
                                if s == test_path:
                                    test_data.setdefault(inp, [])
                                    test_data[inp].append(element.text)
                                    if len(test_data[inp]) == 1:
                                        cats["domain"].append(f"{entry.attrib['category']}")
                                data["rdf"].append(inp)
                                data["text"].append(element.text)
            df = pd.DataFrame(data)
            name = s.split("/")[0]
            if name == "dev":
                name = "validation"
            df.to_csv(
                "webNLG2020_prepared_data/" + name + ".csv", index=False, header=False, sep="\t"
            )
            if s == test_path:
                df_cats = pd.DataFrame(cats)
                df_cats.to_csv(
                "webNLG2020_prepared_data/test_cats.csv", index=False, header=["categories"], sep="\t"
                )
                
                Path("references").mkdir(parents=True, exist_ok=True)
                test = open("webNLG2020_prepared_data/to_predict", "w")
                true = open("webNLG2020_prepared_data/true", "w")
                for i in range(5):
                    with open(f"references/test{i}", "w") as f:
                        for j, k in enumerate(test_data):
                            if len(test_data[k]) - 1 >= i:
                                f.write(test_data[k][i])
                                if i == 0:
                                    test.write(k)
                                    true.write(test_data[k][i])
                            if j != len(test_data) - 1:
                                f.write("\n")
                                if i == 0:
                                    test.write("\n")
                                    true.write("\n")
                test.close()
    else:
        if dataset == "ax":
            save_dir = path
            df = pd.read_csv(path + "all.csv", sep="\t", names=["rdf", "text"])
            categories = ["garden furniture", "white goods", "smartphones", "fashion", "Gartenmöbel", "Weiße Ware", "Mode", "Mobiltelefone"]
            cats_column = list()
            for i in range(len(df)):
                for j in range(len(categories)):
                    if "category➤" + categories[j] in df["rdf"].iloc[i]:
                        cats_column.append(categories[j])
            assert len(cats_column) == len(df)

            df["stratify"] = cats_column

        elif dataset == "webnlg_german":
            sets = [train_path, dev_path]
            save_dir = "gerWebNLG_prepared_data/"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            data = {"rdf": [], "text": [], "stratify": []}
            categories = set()
            for s in sets:
                files = glob.glob(
                    path + "data/v1.6/de/" + s + ".xml", recursive=True
                )
                for file in files:
                    tree = ET.parse(file)
                    root = tree.getroot()
                    for sub_root in root:
                        for entry in sub_root:
                            for element in entry:
                                if "lex" in str(element):
                                    for l in element:
                                        if "sortedtripleset" in str(l):
                                            inp = " && ".join([triple.text for sentence in l for triple in sentence])
                                        if "text" in str(l):
                                            if inp and l.text:
                                                data["rdf"].append(inp)
                                                data["text"].append(l.text)
                                                data["stratify"].append(f"({entry.attrib['size']}, {entry.attrib['category']})")
                                                categories.add(entry.attrib['category'])
            df = pd.DataFrame(data)
            
        # create dictionary where rdf1 : [[text1, text2, text3], stratify_conditions]
        print("num of domains: ", str(len(categories)))
        new = dict()
        for i in range(len(df)):
            new.setdefault(df.iloc[i]["rdf"], [[]])
            new[df.iloc[i]["rdf"]][0].append(df.iloc[i]["text"])
            # stratify hasn't already been added
            if len(new[df.iloc[i]["rdf"]]) == 1:
                new[df.iloc[i]["rdf"]].append(df.iloc[i]["stratify"])
        
        # convert to df
        new_dict = {"rdf": [], "text": [], "stratify": []}
        for k,v in new.items():
            new_dict["rdf"].append(k)
            new_dict["text"].append(v[0])
            new_dict["stratify"].append(v[1])
        new_df = pd.DataFrame(new_dict)
        
        # do splits
        train, dev_test = train_test_split(new_df, test_size=0.3, random_state=8, stratify=new_df["stratify"])
        dev, test = train_test_split(dev_test, test_size=0.5, random_state=2, stratify=dev_test["stratify"])
        
        # convert again to format rdf1 = text1, rdf1 = text2, rdf1 = text3 for final dataframe
        finished_sets = []
        for split in [train, dev, test]:
            split_dict = {"rdf": [], "text": [], "stratify": []}
            for i in range(len(split)):
                for j in range(len(split.iloc[i]["text"])):
                    split_dict["rdf"].append(split.iloc[i]["rdf"])
                    split_dict["text"].append(split.iloc[i]["text"][j])
                    split_dict["stratify"].append(split.iloc[i]["stratify"])
            finished_sets.append(pd.DataFrame(split_dict))
        
        # save sets
        train, dev, test = finished_sets
        for split, name in [(dev[["rdf", "text"]], "validation"), (test[["rdf", "text"]], "test"), (train[["rdf", "text"]], "train")]:
            split.to_csv(
                save_dir + name + ".csv", index=False, header=False, sep="\t"
            )
        
        # create to_predict, true, references/test{i}
        test_data = dict()
        for i in range(len(test)):
            test_data.setdefault(test.iloc[i]["rdf"], [])
            test_data[test.iloc[i]["rdf"]].append(test.iloc[i]["text"])
        
        Path(save_dir + "references").mkdir(parents=True, exist_ok=True)
        test = open(save_dir + "to_predict", "w")
        true = open(save_dir + "true", "w")
        for i in range(5):
            with open(save_dir + f"references/test{i}", "w") as f:
                for j, k in enumerate(test_data):
                    if len(test_data[k]) - 1 >= i:
                        f.write(test_data[k][i])
                        if i == 0:
                            test.write(k)
                            true.write(test_data[k][i])
                    if j != len(test_data) - 1:
                        f.write("\n")
                        if i == 0:
                            test.write("\n")
                            true.write("\n")
        test.close()

if __name__ == "__main__":
    prepare_data()