import tensorflow as tf
import numpy as np

def fgsm_attack(model, x, y, epsilon=0.05):
    """
    Fast Gradient Sign Method (FGSM)
    """

    # Convert inputs to NumPy first (handles Pandas safely)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)

    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)

    x_adv = x + epsilon * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)

    return x_adv.numpy()
