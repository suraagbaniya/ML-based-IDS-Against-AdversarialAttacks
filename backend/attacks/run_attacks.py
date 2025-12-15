import joblib
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from fgsm import fgsm_attack
from pgd import pgd_attack
from jsma import jsma_attack
from attack_evaluator import evaluate_attack

# Load test data
x_test = joblib.load("data/processed/x_test.pkl")
y_test = joblib.load("data/processed/y_test.pkl")


# Load trained neural network
model = tf.keras.models.load_model("backend/ids_model/mlp_model.keras")
print("\nModel and test data loaded successfully")


# Baseline predictions
y_pred_clean = (model.predict(x_test) > 0.5).astype(int)
print("\nBaseline IDS Performance (Clean Data)")
print(classification_report(y_test, y_pred_clean))


# Generate ADVERSARIAL samples
x_adv_fgsm = fgsm_attack(model, x_test, y_test, epsilon=0.05)

# Evaluate
y_pred_fgsm = (model.predict(x_adv_fgsm) > 0.5).astype(int)
print("\nFGSM Attack Performance")
print(classification_report(y_test, y_pred_fgsm))


# PGD Attack Sample
x_adv_pgd = pgd_attack(
    model,
    x_test,
    y_test,
    epsilon=0.05,
    alpha=0.01,
    iterations=10
)

y_pred_pgd = (model.predict(x_adv_pgd) > 0.5).astype(int)

print("\nPGD Attack Performance")
print(classification_report(y_test, y_pred_pgd))


#Quantify Accuracy Drop

fgsm_metrics = evaluate_attack(model, x_test, x_adv_fgsm, y_test)
pgd_metrics = evaluate_attack(model, x_test, x_adv_pgd, y_test)

print("FGSM Metrics:", fgsm_metrics)
print("PGD Metrics:", pgd_metrics)


#Save Adversarial Sample 

joblib.dump(x_adv_fgsm, "data/processed/x_adv_fgsm.pkl")
joblib.dump(x_adv_pgd, "data/processed/x_adv_pgd.pkl")

print("\nAdversarial samples saved for defense training")
