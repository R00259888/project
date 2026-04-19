# (dataset, model, doi, split, eer, auc, acc)
data_from_literature = [
    ("KeystrokeDynamicsBenchmarkDataset", "CNN-LSTM", "\\cite{https://doi.org/10.1109/DICCT64131.2025.10986481}", "80:20", None, 0.960, 0.990),
    # "TABLE I: Proposed Model Result Metrics"

    ("KeyRecs", "KNN", "\\cite{https://doi.org/10.1007/s42452-025-07449-5}", "-", 0.270, None, 0.672),
    ("KeyRecs", "RF", "\\cite{https://doi.org/10.1007/s42452-025-07449-5}", "-", 0.270, None, 0.806),
    ("KeyRecs", "LGBM", "\\cite{https://doi.org/10.1007/s42452-025-07449-5}", "-", 0.200, None, 0.811),
    # "Table 3 Evaluation results for KNN, RF, and LGBM"
    # "Table 4 Mean values for KNN, RF, and LGBM"

    # ("Minecraft-Mouse-Dynamics-Dataset", "RF (Scenario A)", "\\cite{https://doi.org/10.1109/ICECET52533.2021.9698532}", "70:30", 0.001, None, 0.927),
    ("Minecraft-Mouse-Dynamics-Dataset", "RF (Scenario B)", "\\cite{https://doi.org/10.1109/ICECET52533.2021.9698532}", "70:30", 0.396, None, 0.616),
    # TABLE I and II

    ("Mouse-Dynamics-Challenge", "LSTM", "\\cite{https://doi.org/10.48550/arXiv.2504.21415}", "pre-split", 0.0614, 0.9773, None)
    # "TABLE III: User-Averaged Models Performance Comparison on Balabit Dataset"
]

# (dataset, model, doi, split, acc, post_fgsm_acc)
data_from_literature_fgsm = [
    ("CASIA-BIT", "Fingerprint", "\\cite{https://doi.org/10.1111/exsy.13655}", "-", 0.950, 0.731),
    ("CASIA-BIT", "Palmprint", "\\cite{https://doi.org/10.1111/exsy.13655}", "-", 0.998, 0.018),
    ("CASIA-BIT", "Iris", "\\cite{https://doi.org/10.1111/exsy.13655}", "-", 0.995, 0.003)
    # "TABLE 2 Evaluation and comparative analysis of single and multi-biometric user authentication systems."
    # "TABLE 5 Adversarial attacks on triple-biometric user authentication systems with score-level fusion."
]
