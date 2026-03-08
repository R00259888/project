import bob.measure
import numpy as np

from .attacks import impersonation_attack, adversarial_attack

def get_metrics(model, test_dataset, subject_id, attack):
    if attack == "impersonation": test_dataset = impersonation_attack(test_dataset, subject_id)
    X, y_desired = model.prepare_features(test_dataset)
    if attack == "adversarial": X = adversarial_attack(model, X, y_desired)
    confidence_score = model.predict(X).flatten()

    negatives = confidence_score[np.array(y_desired) == 0]
    positives = confidence_score[np.array(y_desired) == 1]

    eer, far, frr, auc = float("nan"), float("nan"), float("nan"), float("nan")

    if not np.isnan(confidence_score).any():
        if len(negatives) > 0 and len(positives) > 0:
            eer_threshold = bob.measure.eer_threshold(negatives, positives)
            far, frr = bob.measure.farfrr(negatives, positives, eer_threshold)
            eer = (far + frr) / 2

        if len(negatives) > 1 and len(positives) > 1:
            curve = bob.measure.roc(negatives, positives, n_points=1000)
            auc = float(np.trapz(np.flip(1 - curve[1]), np.flip(curve[0])))

    return {"eer": eer, "far": far, "frr": frr, "auc": auc}
