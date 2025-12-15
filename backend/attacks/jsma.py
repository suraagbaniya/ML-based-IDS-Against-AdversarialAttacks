import numpy as np

def jsma_attack(model, x, theta=0.1, max_iter=50):
    """
    Simplified Jacobian-based Saliency Map Attack (JSMA)
    """
    x_adv = x.copy()

    for _ in range(max_iter):
        preds = model.predict(x_adv)
        grads = np.gradient(preds, axis=1)

        saliency = np.abs(grads)
        max_idx = np.argmax(saliency, axis=1)

        for i in range(len(x_adv)):
            x_adv[i, max_idx[i]] += theta

    return np.clip(x_adv, 0, 1)
