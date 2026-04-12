# Securing Behavioural-Based Biometrics against Adversarial Attacks

[![Language: Python](https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python&logoColor=white)](https://github.com/R00259888/project/search?l=Python)
[![Language: Jupyter](https://img.shields.io/badge/Language-Jupyter-orange?style=flat-square&logo=jupyter&logoColor=white)](https://github.com/R00259888/project/search?l=Jupyter+Notebook)

This project was made in completion of the AI Research Project module for the [MSc in Artificial Intelligence at MTU Cork](https://www.mtu.ie/courses/crkarti9/).

## Getting Started

### Setup
```bash
git clone https://github.com/R00259888/project && cd project
```

### 1. Running Locally
```bash
chmod +x run.sh && ./run.sh
```

### 2. Running on Google Colab
1. Open Google Drive
2. Upload the project directory to your Google Drive
3. Open the project directory and click `run.ipynb`
4. Click Runtime, then Restart session and run all

Observe the logs, all results can be found in the report directory.
If the experiments fail halfway through due to a timeout or network issue,
they can recover and you can simply rerun and it will continue where it left off.
If Colab RAM usage becomes very high, restarting the session solves the issue, allowing it to continue where the issue occurred.
To reset, delete the data in the report directory and also the cache directory.