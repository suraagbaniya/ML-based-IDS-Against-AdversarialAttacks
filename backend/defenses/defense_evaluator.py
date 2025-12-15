from sklearn.metrics import accuracy_score, f1_score

def evaluate_defense(model, x_test, y_test):
    """
    Evaluate robust IDS performance
    """
    preds = model.predict(x_test)
    preds = (preds > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds)
    }
