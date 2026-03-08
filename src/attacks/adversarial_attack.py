import numpy as np
import tensorflow as tf

from ..utils.fgsm import apply_fgsm_perturbation, padding_mask, FGSM_EPSILON, FGSM_BATCH_SIZE

def adversarial_attack(model, X, y):
    X, y = tf.cast(X, tf.float32), tf.cast(y, tf.float32)

    X_perturbations = []
    for start_index in range(0, X.shape[0], FGSM_BATCH_SIZE):
        X_batch, y_batch = X[start_index:start_index + FGSM_BATCH_SIZE], y[start_index:start_index + FGSM_BATCH_SIZE]

        X_perturbed = apply_fgsm_perturbation(model, X_batch, tf.ones_like(y_batch), padding_mask(X_batch), -1, False)

        not_subject = tf.equal(y_batch, 0.0) # Only apply to instances which are not the subject
        not_subject = tf.reshape(not_subject, [-1, 1, 1])
        X_perturbations.append(tf.where(not_subject, X_perturbed, X_batch).numpy())

    return np.concatenate(X_perturbations, axis=0)
