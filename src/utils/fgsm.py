import tensorflow as tf

FGSM_EPSILON = float(1)
FGSM_BATCH_SIZE = 64

def padding_mask(x):
    return tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1, keepdims=True), tf.float32)

def apply_fgsm_perturbation(model, X, y_desired, mask, direction, training):
    with tf.GradientTape() as tape:
        tape.watch(X)
        y_predicted = model(X, training=training)
        loss = model.compute_loss(y=y_desired, y_pred=y_predicted)

    signed_gradient = tf.sign(tape.gradient(loss, X))
    perturbation = direction * FGSM_EPSILON * signed_gradient * mask

    return X + perturbation
