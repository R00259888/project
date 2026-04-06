import functools, os, random

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .main import get_dataset, get_subject_ids, train_model, train_test_split
from .metrics import get_metrics
from . import comparison_table

vector_variants = [
    {"attack": None, "defence": None, "colour": "red"},

    {"attack": "adversarial", "defence": None, "colour": "blue"},
    {"attack": "adversarial", "defence": "adversarial", "colour": "blue"}
]

time_series_variants = [
    {"attack": None, "defence": None, "colour": "red"},

    {"attack": "impersonation", "defence": None, "colour": "green"},
    {"attack": "impersonation", "defence": "augmentation", "colour": "green"},
    {"attack": "impersonation", "defence": "adversarial", "colour": "green"},

    {"attack": "adversarial", "defence": None, "colour": "blue"},
    {"attack": "adversarial", "defence": "adversarial", "colour": "blue"},
    {"attack": "adversarial", "defence": "augmentation", "colour": "blue"}

]

experiments = [
    {
        "dataset": "IKDD",
        "model": "KeystrokeDynamicsNNModel",
        "subject_count": 5,
        "train_percs": [0.7, 0.8],
        "epochs": 50, # Kamra et al. "Fig. 6: Loss Curve"
        "variants": vector_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        # https://doi.org/10.1109/DICCT64131.2025.10986481
        "dataset": "KeystrokeDynamicsBenchmarkDataset",
        "model": "KeystrokeDynamicsNNModel",
        "subject_count": 5,
        "train_percs": [0.7, 0.8],
        "epochs": 50,
        "variants": vector_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        "dataset": "KeyRecs",
        "model": "LSTMModel",
        "subject_count": 5,
        "train_percs": [0.7, 0.8],
        "epochs": 50,
        "variants": vector_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        # https://doi.org/10.1109/ICECET52533.2021.9698532
        "dataset": "Minecraft-Mouse-Dynamics-Dataset",
        "model": "LSTMModel",
        "subject_count": 5,
        "train_percs": [0.7, 0.8],
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        "dataset": "Minecraft-Mouse-Dynamics-Dataset",
        "model": "CNNLSTMModel",
        "subject_count": 5,
        "train_percs": [0.7, 0.8],
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        # https://doi.org/10.48550/arXiv.2504.21415
        "dataset": "Mouse-Dynamics-Challenge",
        "model": "LSTMModel",
        "subject_count": 5,
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        "dataset": "Mouse-Dynamics-Challenge",
        "model": "CNNLSTMModel",
        "subject_count": 5,
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        "dataset": "Amalgamated-Mouse-Dynamics",
        "model": "LSTMModel",
        "subject_count": 5,
        "train_percs": [0.7],
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        "dataset": "Amalgamated-Mouse-Dynamics",
        "model": "CNNLSTMModel",
        "subject_count": 5,
        "train_percs": [0.7],
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    },
    {
        "dataset": "Amalgamated-Mouse-Dynamics",
        "model": "BlendedLSTMModel",
        "subject_count": 5,
        "train_percs": [0.7],
        "epochs": 50,
        "variants": time_series_variants,
        "plot_metrics": ["eer", "auc", "accuracy"]
    }
]

experiment_legend_handles = [
    matplotlib.patches.Patch(color="red", label="No attack"),
    matplotlib.patches.Patch(color="green", label="Impersonation attack"),
    matplotlib.patches.Patch(color="blue", label="Adversarial attack")
]

@functools.cache
def __subject_ids(dataset):
    return get_subject_ids(get_dataset(dataset))

@functools.cache
def __subject_id_sample(dataset, subject_count):
    subject_ids = __subject_ids(dataset)
    subject_count = min(subject_count, len(subject_ids))
    return random.Random(0).sample(sorted(subject_ids), subject_count)

def __split_label(train_perc):
    if train_perc is None: return "pre-split"
    return f"{int(train_perc * 100)}-{100 - int(train_perc * 100)}"

def __figure_path(name, train_perc):
    return os.path.join("report", "Figures", f"{name}_{__split_label(train_perc)}.png")

def __save_experiment_outputs(name, rows, subject_count, train_perc, plot_metrics):
    name = f"{name}_{__split_label(train_perc)}"
    df = pd.DataFrame(rows)
    tables_path = os.path.join("report", "Tables")
    os.makedirs(tables_path, exist_ok=True)
    df.to_csv(os.path.join(tables_path, name + ".csv"), index=False, float_format="%.3f")

    figures_path = os.path.join("report", "Figures")
    os.makedirs(figures_path, exist_ok=True)
    metric_means = df.groupby(["attack", "defence", "colour"], dropna=False)[plot_metrics].mean().reset_index()

    x_labels, colours = [], []
    for _, row in metric_means.iterrows():
        x_labels.append(f"{row['attack']} vs {row['defence']}")
        colours.append(row["colour"])

    bar_width, bar_spacing = 0.7 / len(plot_metrics), 1.5 / len(plot_metrics)
    bar_positions = np.arange(len(x_labels)) * 1.8
    _, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(plot_metrics):
        bars = ax.bar(bar_positions + (i - (len(plot_metrics) - 1) / 2) * bar_spacing, metric_means[metric], width=bar_width, color=colours)
        labels = []
        for mean in metric_means[metric]:
            labels.append(f"{metric[:3].upper()}\n{mean:.3f}")
        ax.bar_label(bars, labels=labels, padding=3, fontsize=6)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Variant (attack vs defence)")
    ax.set_ylabel("Mean metric value")
    ax.set_title(f"{name}: {subject_count} subjects")
    ax.set_ylim(0, 1.35)
    ax.legend(handles=experiment_legend_handles)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, name + ".png"))
    plt.close()

def __run_variant(dataset, model, attack, defence, subject_ids, train_perc, epochs):
    train, test = train_test_split(get_dataset(dataset), train_perc)
    metrics = []
    for subject_id in subject_ids:
        print(f"{dataset}-{model} {__split_label(train_perc)} attack={attack} defence={defence} subject={subject_id}", flush=True)
        trained_model = train_model(model, subject_id, train, defence, epochs, 0)
        metrics.append((subject_id, get_metrics(trained_model, test, subject_id, attack)))
    return metrics

def main():
    for experiment in experiments:
        dataset = experiment["dataset"]
        model = experiment["model"]
        train_percs = experiment.get("train_percs", [None])

        name = dataset + "_" + model
        subject_count = experiment["subject_count"]
        epochs = experiment["epochs"]
        if subject_count == 0: continue # Skip early
        subject_ids = __subject_id_sample(dataset, subject_count)

        for train_perc in train_percs:
            if os.path.exists(__figure_path(name, train_perc)): continue # Continue where we left off, in case Colab cuts off
            experiment_rows = []
            for variant in experiment["variants"]:
                for subject_id, metrics in __run_variant(dataset, model, variant["attack"], variant["defence"], subject_ids, train_perc, epochs):
                    experiment_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "attack": variant["attack"] or "None",
                        "defence": variant["defence"] or "None",
                        "colour": variant.get("colour") or "red",
                        "subject_id": subject_id,
                        **metrics
                    })

            __save_experiment_outputs(name, experiment_rows, subject_count, train_perc, experiment["plot_metrics"])

    comparison_table.comparison_table()
    overview_chart.overview_chart()

if __name__ == "__main__":
    main()
