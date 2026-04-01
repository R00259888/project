import collections, csv, os

dataset_names = [
    "KeystrokeDynamicsBenchmarkDataset", "IKDD", "KeyRecs",
    "Minecraft-Mouse-Dynamics-Dataset", "Mouse-Dynamics-Challenge",
    "Amalgamated-Mouse-Dynamics"
]

# (dataset, model, doi, split, eer, auc, acc)
data_from_literature = [
    ("KeystrokeDynamicsBenchmarkDataset", "CNN-LSTM", "\\cite{https://doi.org/10.1109/DICCT64131.2025.10986481}", "80:20", None, 0.960, 0.990),
    # "TABLE I: Proposed Model Result Metrics"

    ("KeyRecs", "KNN", "\\cite{https://doi.org/10.1007/s42452-025-07449-5}", "-", 0.270, None, 0.672),
    ("KeyRecs", "RF", "\\cite{https://doi.org/10.1007/s42452-025-07449-5}", "-", 0.270, None, 0.806),
    ("KeyRecs", "LGBM", "\\cite{https://doi.org/10.1007/s42452-025-07449-5}", "-", 0.200, None, 0.811),
    # "Table 3 Evaluation results for KNN, RF, and LGBM"
    # "Table 4 Mean values for KNN, RF, and LGBM"

    ("Minecraft-Mouse-Dynamics-Dataset", "RF (Scenario A)", "\\cite{https://doi.org/10.1109/ICECET52533.2021.9698532}", "70:30", 0.001, None, 0.927),
    ("Minecraft-Mouse-Dynamics-Dataset", "RF (Scenario B)", "\\cite{https://doi.org/10.1109/ICECET52533.2021.9698532}", "70:30", 0.396, None, 0.616),
    # TABLE I and II

    ("Mouse-Dynamics-Challenge", "LSTM", "\\cite{https://doi.org/10.48550/arXiv.2504.21415}", "pre-split", 0.0614, 0.9773, None)
    # "TABLE III: User-Averaged Models Performance Comparison on Balabit Dataset"
]

def __format_eer(eer):
    if eer is None: return "-"
    return f"{eer * 100:.1f}\\%" # Convert to percentage of error

def __format_auc(auc):
    if auc is None: return "-"
    return f"{auc:.3f}"

def __format_acc(acc):
    if acc is None: return "-"
    return f"{acc * 100:.1f}\\%"

def __read_experiment_results():
    tables_dir = os.path.join("report", "Tables")
    rows = collections.defaultdict(list)

    for file_name in sorted(os.listdir(tables_dir)):
        if not file_name.endswith(".csv"): continue

        split = file_name.removesuffix(".csv").rsplit("_", 1)[-1]
        with open(os.path.join(tables_dir, file_name), newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                # Capture the baseline metrics
                if row["attack"] == "None" and row["defence"] == "None":
                    rows[row["dataset"], row["model"], split].append((float(row["eer"]), float(row["auc"]), float(row["accuracy"])))
    experiment_results = {}
    for dataset_model, entries in rows.items():
        # Accumulate the metrics, and gen means
        mean_eer = sum(map(lambda entry: entry[0], entries)) / len(entries)
        mean_auc = sum(map(lambda entry: entry[1], entries)) / len(entries)
        mean_acc = sum(map(lambda entry: entry[2], entries)) / len(entries)
        experiment_results[dataset_model] = (mean_eer, mean_auc, mean_acc)
    return experiment_results

def main():
    experiment_results = __read_experiment_results()

    experiment_results_table = collections.defaultdict(list)
    for (dataset, model, split), (mean_eer, mean_auc, mean_acc) in experiment_results.items():
        experiment_results_table[dataset].append((split, model, mean_eer, mean_auc, mean_acc))

    data_from_literature_table = collections.defaultdict(list)
    for (dataset, model, doi, split, eer, auc, acc) in data_from_literature:
        data_from_literature_table[dataset].append((model, doi, split, eer, auc, acc))

    dataset_groupings = []
    for dataset in dataset_names:
        rows = []
        for (model, doi, split, eer, auc, acc) in data_from_literature_table.get(dataset, []):
            rows.append((model, doi, split, eer, auc, acc))

        for (split, model, mean_eer, mean_auc, mean_acc) in sorted(experiment_results_table.get(dataset, []), key=lambda row: (row[0], row[1])):
            rows.append(("\\texttt{" + model + "}", "This project", split, mean_eer, mean_auc, mean_acc))

        if rows: dataset_groupings.append((dataset, rows))

    comparision_table = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Comparing baseline model performance to the literature (no attack/defence)}",
        "\\label{tab:literaturecomparison}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l l l l r r r}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{Source} & \\textbf{Split} & \\textbf{EER} & \\textbf{AUC} & \\textbf{ACC} \\\\",
        "\\midrule",
    ]

    for grouping_index, (dataset, rows) in enumerate(dataset_groupings):
        for row_index, (model, doi, split, eer, auc, acc) in enumerate(rows):
            if row_index == 0: __dataset = dataset
            else: __dataset = ""

            if row_index == len(rows) - 1: end_of_line = "\\\\[4pt]"
            else: end_of_line = "\\\\"

            comparision_table.append(f"{__dataset} & {model} & {doi} & {split} & {__format_eer(eer)} & {__format_auc(auc)} & {__format_acc(acc)} {end_of_line}")

        if grouping_index == len(dataset_groupings) - 1: comparision_table.append("\\bottomrule")
        else: comparision_table.append("\\midrule")

    output_file = os.path.join("report", "Tables", "comparison_table.tex")

    comparision_table += ["\\end{tabular}%", "}", "\\end{table}", ""]
    with open(output_file, "w") as csv_file: csv_file.write("\n".join(comparision_table))

if __name__ == "__main__":
    main()