import collections, glob, os

import pandas as pd

from . import literature

dataset_names = [
    "KeystrokeDynamicsBenchmarkDataset", "IKDD", "KeyRecs",
    "Minecraft-Mouse-Dynamics-Dataset", "Mouse-Dynamics-Challenge",
    "Amalgamated-Mouse-Dynamics",
]

def read_all():
    dataframes = []
    for path in sorted(glob.glob(os.path.join("report", "Tables", "*.csv"))):
        split = os.path.basename(path).removesuffix(".csv").rsplit("_", 1)[-1]
        dataframe = pd.read_csv(path, keep_default_na=False)
        for metric in ["eer", "far", "frr", "auc", "accuracy", "precision", "recall"]:
            if metric in dataframe.columns: dataframe[metric] = pd.to_numeric(dataframe[metric], errors="coerce")
        for column in ["attack", "defence"]:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].replace("", "None")
        dataframe["split"] = split
        dataframes.append(dataframe)

    if dataframes: return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()

def __read_grouped_attack_defence_dataframe(attack, defence):
    dataframe = read_all()
    filtered = dataframe[(dataframe["attack"] == attack) & (dataframe["defence"] == defence)]
    return filtered.groupby(["dataset", "model", "split"], as_index=False)[["eer", "auc", "accuracy"]].mean()

def read_literature_dataframe():
    return pd.DataFrame(literature.data_from_literature, \
        columns=["dataset", "model", "source", "split", "eer", "auc", "accuracy"])

def read_fgsm_literature_dataframe():
    return pd.DataFrame(literature.data_from_literature_fgsm,
        columns=["dataset", "model", "source", "split", "accuracy", "baseline_accuracy"])

def read_fgsm_dataframe():
    return __read_grouped_attack_defence_dataframe("adversarial", "None")

def read_baseline_dataframe(): # Capture the baseline metrics
    return __read_grouped_attack_defence_dataframe("None", "None")

def read_baseline():
    baseline = read_baseline_dataframe()
    baseline_results = {}
    for row in baseline.itertuples(index=False):
        baseline_results[(row.dataset, row.model, row.split)] = (row.eer, row.auc, row.accuracy)
    return baseline_results

def read_baseline_grouped_by_dataset():
    results_by_dataset = collections.defaultdict(list)
    baseline = read_baseline_dataframe().sort_values(["dataset", "split", "model"])
    for row in baseline.itertuples(index=False):
        results_by_dataset[row.dataset].append((row.split, row.model, row.eer, row.auc, row.accuracy))
    return results_by_dataset

def best_baseline_eer_by_dataset():
    return read_baseline_dataframe().groupby("dataset")["eer"].min().to_dict()

def best_literature_eer_by_dataset():
    literature_df = read_literature_dataframe()
    literature_df = literature_df[literature_df["eer"].notna()]

    best_eer_row_indices = literature_df.groupby("dataset")["eer"].idxmin()
    best_eer_rows = literature_df.loc[best_eer_row_indices, ["dataset", "eer", "model"]]
    return {row.dataset: (row.eer, row.model) for row in best_eer_rows.itertuples(index=False)}
