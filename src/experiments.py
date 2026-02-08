import os, subprocess

experiments = [
    {"dataset": "IKDD", "model": "KeystrokeDynamicsNNModel", "subject_id": 17},
    {"dataset": "Minecraft-Mouse-Dynamics-Dataset", "model": "MouseDynamicsLSTMModel", "subject_id": 0},
    {"dataset": "Minecraft-Mouse-Dynamics-Dataset", "model": "MouseDynamicsCNNAndLSTMModel", "subject_id": 0},
]

def main():
    for experiment in experiments:
        subprocess_run_args = [
            "python3", "-m", "src.main",

            "--dataset", experiment["dataset"],
            "--evaluation_plot", os.path.join("thesis", "Figures", experiment["model"] + "_confidence.png"),
            "--model", experiment["model"],
            "--subject_id", str(experiment["subject_id"])
        ]  
        subprocess.run(subprocess_run_args)

if __name__ == "__main__":
    main()
