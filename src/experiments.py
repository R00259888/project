import os, subprocess

experiments = [
    {"dataset": "IKDD", "model": "KeystrokeDynamicsNNModel", "subject_id": 17},

    {"dataset": "Minecraft-Mouse-Dynamics-Dataset", "model": "MouseDynamicsLSTMModel", "subject_id": 0},
    {"dataset": "Minecraft-Mouse-Dynamics-Dataset", "model": "MouseDynamicsCNNAndLSTMModel", "subject_id": 0},

    {"attack": "None", "defence": "defensive_model", "dataset": "Mouse-Dynamics-Challenge", "model": "MouseDynamicsLSTMModel", "subject_id": 21},
    {"attack": "None", "defence": "None", "dataset": "Mouse-Dynamics-Challenge", "model": "MouseDynamicsLSTMModel", "subject_id": 21},
    {"attack": "erratic", "defence": "defensive_model", "dataset": "Mouse-Dynamics-Challenge", "model": "MouseDynamicsLSTMModel", "subject_id": 21},
    {"attack": "erratic", "defence": "None", "dataset": "Mouse-Dynamics-Challenge", "model": "MouseDynamicsLSTMModel", "subject_id": 21}
]

def main():
    for experiment in experiments:
        attack = "None" if "attack" not in experiment else experiment["attack"]
        defence = "None" if "defence" not in experiment else experiment["defence"]
        attack_vs_defense = attack.capitalize() + "_vs_" + defence.capitalize()

        dataset = experiment["dataset"]
        evaluation_parameters = [dataset.replace("-", ""), experiment["model"], attack_vs_defense]
        evaluation_plot = "_".join(evaluation_parameters) + "_confidence.png"
        evaluation_plot = os.path.join("thesis", "Figures", evaluation_plot)

        subprocess_run_args = [
            "python3", "-m", "src.main",

            "--attack", attack,
            "--defence", defence,
            "--dataset", dataset,
            "--evaluation_plot", evaluation_plot,
            "--model", experiment["model"],
            "--subject_id", str(experiment["subject_id"]),
        ]
        subprocess.run(subprocess_run_args)

if __name__ == "__main__":
    main()
