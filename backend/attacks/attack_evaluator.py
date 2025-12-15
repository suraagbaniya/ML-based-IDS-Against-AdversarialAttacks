from sklearn.metrics import accuracy_score

def evaluate_attack(model, x_clean, x_adv, y):
    """
    Compare clean vs adversarial accuracy
    """
    clean_pred = (model.predict(x_clean) > 0.5).astype(int)
    adv_pred = (model.predict(x_adv) > 0.5).astype(int)

    clean_acc = accuracy_score(y, clean_pred)
    adv_acc = accuracy_score(y, adv_pred)

    return {
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "accuracy_drop": clean_acc - adv_acc
    }
