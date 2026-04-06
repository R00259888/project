import os

import matplotlib.pyplot as plt
import numpy as np

from . import experiment_results

def __build_row_labels(dataframe):
    return dataframe["dataset"] + "_" + dataframe["model"] + "_" + dataframe["split"]

def __build_scenario_labels(dataframe):
    return dataframe["attack"] + " vs " + dataframe["defence"]

def __label_heatmap(ax, matrix):
    for i, row in enumerate(matrix):
        for j, eer_value in enumerate(row):
            if np.isnan(eer_value): continue
            ax.text(j, i, f"{eer_value:.3f}", ha="center", va="center", fontsize=7, color="black")

def __label_bars(ax, bars, labels=None):
    if labels is None: labels = [""] * len(bars)

    for bar, label in zip(bars, labels):
        eer_value = bar.get_width()
        if np.isnan(eer_value): continue

        model_label_suffix = ""
        if label: model_label_suffix = f" ({label})"
        ax.text(eer_value, bar.get_y() + bar.get_height() / 2, f"{eer_value:.1%}{model_label_suffix}", va="center", fontsize=7)

def __build_literature_comparison():
    project_eer_by_dataset = experiment_results.best_baseline_eer_by_dataset()
    literature_by_dataset = experiment_results.best_literature_eer_by_dataset()

    has_project_and_literature_results = lambda dataset: dataset in project_eer_by_dataset and dataset in literature_by_dataset
    datasets = list(filter(has_project_and_literature_results, experiment_results.dataset_names))
    project_eer_values = [project_eer_by_dataset[dataset] for dataset in datasets]
    literature_eer_values = [literature_by_dataset[dataset][0] for dataset in datasets]
    literature_model_labels = [literature_by_dataset[dataset][1] for dataset in datasets]

    return datasets, project_eer_values, literature_eer_values, literature_model_labels

def __build_experiment_configurations(experiment_dataframe):
    dataset_order = {dataset_name: index for index, dataset_name in enumerate(experiment_results.dataset_names)}

    experiment_configurations = experiment_dataframe[["dataset", "model", "split"]] \
        .drop_duplicates() \
        .assign(_order=lambda frame: frame["dataset"].map(dataset_order).fillna(len(dataset_order))) \
        .sort_values(["_order", "model", "split"]) \
        .drop(columns="_order") \
        .reset_index(drop=True)
    
    experiment_configurations["row_label"] = __build_row_labels(experiment_configurations)
    return experiment_configurations

def __build_heatmap(experiment_dataframe, experiment_configurations, scenario_labels):
    mean_eer = experiment_dataframe.groupby(["dataset", "model", "split", "attack", "defence"], as_index=False)["eer"].mean()
    mean_eer["row_label"] = __build_row_labels(mean_eer)
    mean_eer["scenario_label"] = __build_scenario_labels(mean_eer)

    heatmap_frame = mean_eer.pivot(index="row_label", columns="scenario_label", values="eer")
    heatmap_frame = heatmap_frame.reindex(index=experiment_configurations["row_label"], columns=scenario_labels)

    row_labels = experiment_configurations["row_label"].to_list()
    heatmap_matrix = heatmap_frame.to_numpy()

    dataset_boundary_row_indices = np.flatnonzero(experiment_configurations["dataset"].ne(experiment_configurations["dataset"].shift()).to_numpy())
    dataset_row_separators = (dataset_boundary_row_indices[1:] - 0.5).tolist()

    return heatmap_matrix, row_labels, dataset_row_separators

def __build_scenarios(experiment_dataframe):
    scenarios_dataframe = experiment_dataframe[["attack", "defence", "colour"]].drop_duplicates().reset_index(drop=True)
    scenarios_dataframe["scenario_label"] = __build_scenario_labels(scenarios_dataframe)

    scenario_labels = scenarios_dataframe["scenario_label"].to_list()
    scenario_colours = scenarios_dataframe["colour"].fillna("black").to_list()

    attack_change_indices = scenarios_dataframe.index[scenarios_dataframe["attack"].ne(scenarios_dataframe["attack"].shift())].to_list()
    scenario_separators = [index - 0.5 for index in attack_change_indices[1:]]

    return scenario_labels, scenario_colours, scenario_separators

def __build_figure():
    figure = plt.figure(figsize=(14, 9), layout="constrained")
    grid_spec = figure.add_gridspec(2, 1)

    heatmap_axis = figure.add_subplot(grid_spec[0, 0])
    literature_axis = figure.add_subplot(grid_spec[1, 0])
    return figure, heatmap_axis, literature_axis

def __plot_heatmap(ax, heatmap_matrix):
    heatmap_cmap = plt.get_cmap("RdYlGn").copy()
    heatmap_image = ax.imshow(heatmap_matrix, cmap=heatmap_cmap, vmin=0, vmax=1, aspect="auto")
    __label_heatmap(ax, heatmap_matrix)
    return heatmap_image

def __style_heatmap(ax, figure, heatmap_image, row_labels, dataset_row_separators, scenario_labels, scenario_colours, scenario_separators):
    ax.set_xticks(range(len(scenario_labels)))
    scenario_tick_labels = ax.set_xticklabels(scenario_labels, fontsize=7, rotation=45, ha="right")
    for label, colour in zip(scenario_tick_labels, scenario_colours): label.set_color(colour)

    ax.set_yticks(range(len(row_labels))) # Y labels for heatmap
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Attack vs Defence", fontsize=8)
    ax.set_title("Mean EER for each experiment and attack/defence scenario (green is better, grey N/A)", fontsize=9, pad=4)

    colourbar = figure.colorbar(heatmap_image, ax=ax)
    colourbar.set_label("EER", fontsize=8)

def __plot_literature_comparison(ax):
    datasets, project_eer_values, literature_eer_values, literature_model_labels = __build_literature_comparison()

    y_positions, bar_height = np.arange(len(datasets)), 0.25
    project_bars = ax.barh(y_positions + bar_height / 2, project_eer_values, bar_height, color="blue", label="This project")
    literature_bars = ax.barh(y_positions - bar_height / 2, literature_eer_values, bar_height, color="red", label="Literature")

    __label_bars(ax, project_bars)
    __label_bars(ax, literature_bars, literature_model_labels)

    ax.set_yticks(y_positions) # Y labels for literature comparision
    ax.set_yticklabels(datasets, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("EER (lower better)", fontsize=8)
    ax.set_title("Baseline vs Literature (no attack/defence)", fontsize=9, pad=6)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(axis="x")

def adversarial_summary_chart():
    figures_dir = os.path.join("report", "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    experiment_dataframe = experiment_results.read_all()
    experiment_configurations = __build_experiment_configurations(experiment_dataframe)
    scenario_labels, scenario_colours, scenario_separators = __build_scenarios(experiment_dataframe)
    heatmap_matrix, row_labels, dataset_row_separators = __build_heatmap(experiment_dataframe, experiment_configurations, scenario_labels)

    figure, heatmap_axis, literature_axis = __build_figure()
    heatmap_image = __plot_heatmap(heatmap_axis, heatmap_matrix)
    __style_heatmap(heatmap_axis, figure, heatmap_image, row_labels, dataset_row_separators, scenario_labels, scenario_colours, scenario_separators)

    __plot_literature_comparison(literature_axis)

    plt.savefig(os.path.join(figures_dir, "adversarial_summary_chart.png"))
    plt.close()

def main():
    adversarial_summary_chart()

if __name__ == "__main__":
    main()
