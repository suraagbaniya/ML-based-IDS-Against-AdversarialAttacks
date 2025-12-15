import joblib
import tensorflow as tf
import numpy as np

from metrics import compute_metrics
from plots import plot_confusion, plot_roc, plot_accuracy_comparison

# Load data
x_test = joblib.load("data/processed/x_test.pkl")
y_test = joblib.load("data/processed/y_test.pkl")
x_adv_fgsm = joblib.load("data/processed/x_adv_fgsm.pkl")
x_adv_pgd = joblib.load("data/processed/x_adv_pgd.pkl")

# Load models
mlp = tf.keras.models.load_model("backend/ids_model/mlp_model.keras")
mlp_robust = tf.keras.models.load_model("backend/ids_model/mlp_robust.keras")
ensemble = joblib.load("backend/ids_model/ensemble.pkl")

print("Evaluation setup loaded")

# Evaluate Clean vs Adversarial (Before Defense)
# Clean predictions
y_prob_clean = mlp.predict(x_test).ravel()
y_pred_clean = (y_prob_clean > 0.5).astype(int)

# FGSM predictions
y_prob_fgsm = mlp.predict(x_adv_fgsm).ravel()
y_pred_fgsm = (y_prob_fgsm > 0.5).astype(int)

print("MLP Clean Metrics:", compute_metrics(
    y_test, y_pred_clean, y_prob_clean))
print("MLP FGSM Metrics:", compute_metrics(y_test, y_pred_fgsm, y_prob_fgsm))

plot_confusion(y_test, y_pred_clean, "MLP - Clean Data")
plot_confusion(y_test, y_pred_fgsm, "MLP - FGSM Attack")

plot_roc(y_test, y_prob_clean, "MLP ROC - Clean")
plot_roc(y_test, y_prob_fgsm, "MLP ROC - FGSM")


# Evaluate After Defense (Robust Model)
# Robust model predictions
y_prob_robust = mlp_robust.predict(x_adv_fgsm).ravel()
y_pred_robust = (y_prob_robust > 0.5).astype(int)

print("Robust MLP Metrics:", compute_metrics(
    y_test, y_pred_robust, y_prob_robust))

plot_confusion(y_test, y_pred_robust, "Robust MLP - FGSM")
plot_roc(y_test, y_prob_robust, "Robust MLP ROC - FGSM")


# Evaluate Ensemble Defense
ensemble_preds = ensemble.predict(x_test)
ensemble_probs = ensemble.predict_proba(x_test)[:, 1]

print("Ensemble Metrics:", compute_metrics(
    y_test, ensemble_preds, ensemble_probs))

plot_confusion(y_test, ensemble_preds, "Ensemble IDS - Clean Data")
plot_roc(y_test, ensemble_probs, "Ensemble IDS ROC")


# Accuracy Comparison Plot
accuracies = [
    compute_metrics(y_test, y_pred_clean)["accuracy"],
    compute_metrics(y_test, y_pred_fgsm)["accuracy"],
    compute_metrics(y_test, y_pred_robust)["accuracy"],
    compute_metrics(y_test, ensemble_preds)["accuracy"]
]

labels = [
    "MLP Clean",
    "MLP FGSM",
    "Robust MLP FGSM",
    "Ensemble Clean"
]

plot_accuracy_comparison(labels, accuracies)
