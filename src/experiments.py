import functools, os, random

import matplotlib.pyplot as plt
import pandas as pd

from .main import get_metrics, get_dataset, get_subject_ids, train_model, train_test_split

experiments = [
    {
        "dataset": "IKDD",
        "model": "KeystrokeDynamicsNNModel",
        "subject_count": 3,
        "variants": [{"attack": None, "defence": None}]
    },
    {
        "dataset": "Minecraft-Mouse-Dynamics-Dataset",
        "model": "MouseDynamicsLSTMModel",
        "subject_count": 3,
        "variants": [{"attack": None, "defence": None}]
    },
    {
        "dataset": "Minecraft-Mouse-Dynamics-Dataset",
        "model": "MouseDynamicsCNNAndLSTMModel",
        "subject_count": 3,
        "variants": [{"attack": None, "defence": None}]
    },
    {
        "dataset": "Mouse-Dynamics-Challenge",
        "model": "MouseDynamicsLSTMModel",
        "subject_count": 3,
        "variants": [
            {"attack": None, "defence": None},
            {"attack": None, "defence": "augmentation"},
            {"attack": "impersonation", "defence": None},
            {"attack": "impersonation", "defence": "augmentation"}
        ]
    }
]

@functools.cache
def __subject_ids(dataset):
    return get_subject_ids(get_dataset(dataset))

@functools.cache
def __subject_id_sample(dataset, subject_count):
    subject_ids = __subject_ids(dataset)
    subject_count = min(subject_count, len(subject_ids))
    return random.Random(0).sample(sorted(subject_ids), subject_count)

def __save_experiment_outputs(name, rows):
    df = pd.DataFrame(rows)
    tables_path = os.path.join("thesis", "Tables")
    os.makedirs(tables_path, exist_ok=True)
    df.to_csv(os.path.join(tables_path, name + ".csv"), index=False, float_format="%.3f")

    figures_path = os.path.join("thesis", "Figures")
    os.makedirs(figures_path, exist_ok=True)
    metric_means = (df.groupby(["attack", "defence"], dropna=False)["eer"].mean().reset_index())

    x_labels = []
    for _, row in metric_means.iterrows():
        x_labels.append(f"{row['attack']} vs {row['defence']}")

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_labels, metric_means["eer"], width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Mean EER")
    ax.set_title(name)
    ax.set_ylim(0, 1.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, name + ".png"))
    plt.close()

def __run_variant(dataset, model, attack, defence, subject_ids):
    train, test = train_test_split(get_dataset(dataset))
    metrics = []
    for subject_id in subject_ids:
        trained_model = train_model(model, subject_id, train, defence, 20, 0)
        metrics.append((subject_id, get_metrics(trained_model, test, subject_id, attack)))
    return metrics

def main():
    for experiment in experiments:
        dataset = experiment["dataset"]
        model = experiment["model"]

        subject_count = experiment["subject_count"]
        if subject_count == 0: continue # Skip early
        subject_ids = __subject_id_sample(dataset, subject_count)

        experiment_rows = []
        for variant in experiment["variants"]:
            for subject_id, metrics in __run_variant(dataset, model, variant["attack"], variant["defence"], subject_ids):
                experiment_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "attack": variant["attack"] or "None",
                    "defence": variant["defence"] or "None",
                    "subject_id": subject_id,
                    **metrics
                })

        name = dataset + "_" + model
        __save_experiment_outputs(name, experiment_rows)

if __name__ == "__main__":
    main()
