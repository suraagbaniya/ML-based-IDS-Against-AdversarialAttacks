import tensorflow as tf
import numpy as np

def pgd_attack(model, x, y, epsilon=0.05, alpha=0.01, iterations=10):
    """
    Projected Gradient Descent (PGD)
    """

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    x_adv = tf.identity(x)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            predictions = model(x_adv)
            loss = tf.keras.losses.binary_crossentropy(y, predictions)

        gradient = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(gradient)

        perturbation = tf.clip_by_value(x_adv - x, -epsilon, epsilon)
        x_adv = tf.clip_by_value(x + perturbation, 0, 1)

    return x_adv.numpy()
