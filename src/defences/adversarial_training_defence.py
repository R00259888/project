import tensorflow as tf

from ..utils.fgsm import apply_fgsm_perturbation, padding_mask

def adversarial_defence(model, dataset):
    X, y = dataset[0], tf.cast(dataset[1], tf.float32)
    if len(dataset) > 2: sample_weight = dataset[2]
    else: sample_weight = None

    X_perturbed = apply_fgsm_perturbation(model, X, y, padding_mask(X), 1, True)

    with tf.GradientTape() as tape:
        losses = [
            model.compute_loss(y=y, y_pred=model(X, training=True), sample_weight=sample_weight),
            model.compute_loss(y=y, y_pred=model(X_perturbed, training=True), sample_weight=sample_weight)
        ]
        loss = tf.reduce_mean(losses)

    model.optimizer.apply(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
    return {"loss": loss}
