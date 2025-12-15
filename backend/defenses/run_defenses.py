import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from backend.defenses.adversarial_training import adversarial_training
from backend.attacks.fgsm import fgsm_attack
from backend.defenses.feature_squeezing import bit_depth_reduction
from backend.defenses.input_denoising import gaussian_denoising
from backend.defenses.ensemble_defense import build_ensemble
from backend.defenses.defense_evaluator import evaluate_defense

# ----------------------------
# LOAD DATA
# ----------------------------
x_train = joblib.load("data/processed/x_train.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
x_test = joblib.load("data/processed/x_test.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Load neural IDS model
mlp = tf.keras.models.load_model("backend/ids_model/mlp_model.keras")

print("All data and models loaded successfully")

# ----------------------------
# DEFENSE 1: ADVERSARIAL TRAINING
# ----------------------------
mlp_robust = adversarial_training(
    model=mlp,
    x_train=x_train,
    y_train=y_train,
    attack_fn=fgsm_attack,
    epochs=3,          # keep small, IDS data is large
    batch_size=1024,
    epsilon=0.05
)

mlp_robust.save("backend/ids_model/mlp_robust.keras")
print("Adversarial training completed")

# ----------------------------
# DEFENSE 2: FEATURE SQUEEZING
# ----------------------------
x_test_squeezed = bit_depth_reduction(x_test, bits=8)
print("Feature squeezing applied")

# ----------------------------
# DEFENSE 3: INPUT DENOISING
# ----------------------------
x_test_denoised = gaussian_denoising(x_test, sigma=0.5)
print("Input denoising applied")

# ----------------------------
# DEFENSE 4: ENSEMBLE DEFENSE
# ----------------------------
rf = joblib.load("backend/ids_model/baseline_rf.pkl")
xgb = joblib.load("backend/ids_model/baseline_xgb.pkl")

ensemble = build_ensemble(rf, xgb)
ensemble.fit(x_train, y_train)

joblib.dump(ensemble, "backend/ids_model/ensemble.pkl")
print("Ensemble defense model created")

# ----------------------------
# EVALUATION
# ----------------------------
robust_results = evaluate_defense(mlp_robust, x_test, y_test)

ensemble_preds = ensemble.predict(x_test)
ensemble_results = {
    "accuracy": accuracy_score(y_test, ensemble_preds),
    "f1_score": f1_score(y_test, ensemble_preds)
}

print("\nRobust MLP Results:", robust_results)
print("Ensemble Results:", ensemble_results)
