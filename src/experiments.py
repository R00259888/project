import os, subprocess

experiments = [
    {"model": "keystroke", "subject_id": 17},
    {"model": "mouse", "subject_id": 0}
]

def main():
    for experiment in experiments:
        subprocess_run_args = [
            "python3", "-m", "src.main",

            "--model", experiment["model"],
            "--subject_id", str(experiment["subject_id"]),
            "--evaluation_plot", os.path.join("thesis", "Figures", experiment["model"] + "_confidence.png")
        ]  
        subprocess.run(subprocess_run_args)

if __name__ == "__main__":
    main()
