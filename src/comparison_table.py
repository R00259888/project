import os

import pandas as pd

from . import experiment_results

def __format_eer(eer):
    if pd.isna(eer): return "-"
    return f"{float(eer) * 100:.1f}\\%" # Convert to percentage of error

def __format_auc(auc):
    if pd.isna(auc): return "-"
    return f"{float(auc):.3f}"

def __format_acc(acc):
    if pd.isna(acc): return "-"
    return f"{float(acc) * 100:.1f}\\%"

def __format_delta_acc(delta):
    if pd.isna(delta): return "-"
    sign = ""
    if delta >= 0: sign = "+"
    return f"{sign}{float(delta) * 100:.1f}\\%"

def __sort_frame(dataframe):
    dataset_names = list(experiment_results.dataset_names)
    for dataset in dataframe["dataset"].unique():
        if dataset not in dataset_names: dataset_names.append(dataset)

    dataframe["dataset"] = pd.Categorical(dataframe["dataset"], categories=dataset_names, ordered=True)
    return dataframe.sort_values(["dataset", "source_order"]).copy()

def __build_baseline_table_dataframe():
    literature_dataframe = experiment_results.read_literature_dataframe()
    literature_dataframe["source_order"] = 0

    project_dataframe = experiment_results.read_baseline_dataframe().copy()
    project_dataframe["model"] = project_dataframe["model"].map(lambda model_name: f"\\texttt{{{model_name}}}")
    project_dataframe["source"] = "This project"
    project_dataframe["source_order"] = 1

    project_columns = ["dataset", "model", "source", "split", "eer", "auc", "accuracy", "source_order"]
    table_dataframe = pd.concat([literature_dataframe, project_dataframe[project_columns]], ignore_index=True)
    table_dataframe = __sort_frame(table_dataframe)

    table_dataframe["EER"] = table_dataframe["eer"].map(__format_eer)
    table_dataframe["AUC"] = table_dataframe["auc"].map(__format_auc)
    table_dataframe["ACC"] = table_dataframe["accuracy"].map(__format_acc)

    return table_dataframe[["dataset", "model", "source", "split", "EER", "AUC", "ACC"]]

def __build_fgsm_table_dataframe():
    literature_dataframe = experiment_results.read_fgsm_literature_dataframe()
    literature_dataframe["source_order"] = 0

    fgsm_dataframe = experiment_results.read_fgsm_dataframe().copy()
    baseline_source_dataframe = experiment_results.read_baseline_dataframe()
    baseline_columns = ["dataset", "model", "split", "accuracy"]
    baseline_accuracy_dataframe = baseline_source_dataframe[baseline_columns]
    baseline_dataframe = baseline_accuracy_dataframe.rename(columns={"accuracy": "baseline_acc"})

    fgsm_dataframe = fgsm_dataframe.merge(baseline_dataframe, on=["dataset", "model", "split"], how="left")
    fgsm_dataframe["model"] = fgsm_dataframe["model"].map(lambda model_name: f"\\texttt{{{model_name}}}")
    fgsm_dataframe["source"] = "This project"
    fgsm_dataframe["source_order"] = 1

    fgsm_columns = ["dataset", "model", "source", "split", "accuracy", "baseline_acc", "source_order"]
    table_dataframe = pd.concat([literature_dataframe, fgsm_dataframe[fgsm_columns]], ignore_index=True)
    table_dataframe = __sort_frame(table_dataframe)

    table_dataframe["ACC"] = table_dataframe["baseline_acc"].map(__format_acc)
    table_dataframe["Post-FGSM ACC"] = table_dataframe["accuracy"].map(__format_acc)
    table_dataframe["$\\Delta$ ACC"] = (table_dataframe["accuracy"] - table_dataframe["baseline_acc"]).map(__format_delta_acc)

    return table_dataframe[["dataset", "model", "source", "split", "ACC", "Post-FGSM ACC", "$\\Delta$ ACC"]]

def __insert_dataset_midrules(tabular, datasets):
    table_lines = tabular.splitlines()
    header_midrule_index = table_lines.index("\\midrule")
    bottomrule_index = table_lines.index("\\bottomrule")
    table_body_lines = table_lines[header_midrule_index + 1: bottomrule_index]

    lines_with_midrule_added = []
    previous_dataset = None
    for row_index, row_line in enumerate(table_body_lines):
        current_dataset = datasets.iloc[row_index]
        if row_index > 0 and current_dataset != previous_dataset:lines_with_midrule_added.append("\\midrule")
        lines_with_midrule_added.append(row_line)
        previous_dataset = current_dataset

    table_lines[header_midrule_index + 1:bottomrule_index] =  lines_with_midrule_added
    return "\n".join(table_lines)

def __write_comparison_table(dataframe, filename, caption, label):
    display_headers = ["Dataset", "Model", "Source", "Split", *dataframe.columns[4:]] #Extract everything after first 4 cols
    latex_tabular = dataframe.to_latex(index=False, escape=False, column_format="l l l l  r r r", header=display_headers)
    latex_tabular = __insert_dataset_midrules(latex_tabular, dataframe["dataset"])

    table_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        latex_tabular.rstrip(),
        "}",
        "\\end{table}",
        "",
    ]

    with open(os.path.join("report", "Tables", filename), "w") as tex_file:
        tex_file.write("\n".join(table_lines))

def comparison_table():
    __write_comparison_table(__build_baseline_table_dataframe(), "comparison_table.tex", \
        "Compare the baseline model performance against the literature (without attack/defence)", "tab:literaturecomparison")
    __write_comparison_table(__build_fgsm_table_dataframe(), "fgsm_comparison_table.tex", \
        "Compare the post-FGSM model performance against the literature", "tab:fgsmcomparison")

def main():
    comparison_table()

if __name__ == "__main__":
    main()