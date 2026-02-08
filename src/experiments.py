import os, subprocess

experiments = [
    {"dataset": "IKDD", "model": "KeystrokeDynamicsNNModel", "subject_id": 17},

    {"dataset": "Minecraft-Mouse-Dynamics-Dataset", "model": "MouseDynamicsLSTMModel", "subject_id": 0},
    {"dataset": "Minecraft-Mouse-Dynamics-Dataset", "model": "MouseDynamicsCNNAndLSTMModel", "subject_id": 0},

    {"attack": "erratic", "dataset": "Mouse-Dynamics-Challenge", "model": "MouseDynamicsLSTMModel", "subject_id": 7},
    {"attack": "None", "dataset": "Mouse-Dynamics-Challenge", "model": "MouseDynamicsLSTMModel", "subject_id": 7}
]

def main():
    for experiment in experiments:
        attack = "None" if "attack" not in experiment else experiment["attack"]
        dataset = experiment["dataset"]
        evaluation_parameters = [dataset.replace("-", ""), experiment["model"], attack.capitalize()]
        evaluation_plot = "_".join(evaluation_parameters) + "_confidence.png"
        evaluation_plot = os.path.join("thesis", "Figures", evaluation_plot)

        subprocess_run_args = [
            "python3", "-m", "src.main",

            "--attack", attack,
            "--dataset", dataset,
            "--evaluation_plot", evaluation_plot,
            "--model", experiment["model"],
            "--subject_id", str(experiment["subject_id"]),
        ]
        subprocess.run(subprocess_run_args)

if __name__ == "__main__":
    main()
