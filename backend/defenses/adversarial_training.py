import numpy as np
import tensorflow as tf

def adversarial_training(
    model,
    x_train,
    y_train,
    attack_fn,
    epochs=5,
    batch_size=1024,
    epsilon=0.05
):
    """
    Memory-safe adversarial training using mini-batches
    """

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(batch_size)

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}] Adversarial Training")

        for step, (x_batch, y_batch) in enumerate(dataset):
            # Generate adversarial samples for this batch ONLY
            x_adv_batch = attack_fn(
                model,
                x_batch,
                y_batch,
                epsilon=epsilon
            )

            # Combine clean + adversarial batch
            x_combined = tf.concat([x_batch, x_adv_batch], axis=0)
            y_combined = tf.concat([y_batch, y_batch], axis=0)

            # Train on combined batch
            model.train_on_batch(x_combined, y_combined)

        print(f"Epoch {epoch+1} completed")

    return model
