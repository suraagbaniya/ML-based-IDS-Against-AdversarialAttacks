from django.shortcuts import render
import os
import numpy as np
import tensorflow as tf
import joblib

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response

# =====================================
# Load models using ABSOLUTE PATHS (ONCE)
# =====================================

BASE_DIR = settings.BASE_DIR  # backend/ids_backend

MLP_MODEL_PATH = os.path.join(BASE_DIR, "..", "ids_model", "mlp_robust.keras")
ENSEMBLE_MODEL_PATH = os.path.join(BASE_DIR, "..", "ids_model", "ensemble.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "scaler.pkl")


# Optional debug (remove later)
print("Loading MLP model from:", MLP_MODEL_PATH)
print("Loading ensemble model from:", ENSEMBLE_MODEL_PATH)
print("Loading scaler from:", SCALER_PATH)

mlp_model = tf.keras.models.load_model(MLP_MODEL_PATH)
ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# =========================
# Prediction API
# =========================
class PredictAPIView(APIView):
    def post(self, request):
        try:
            features = request.data.get("features")

            if features is None:
                return Response(
                    {"error": "No features provided"},
                    status=400
                )

            x = np.array(features)
            x = scaler.transform(x)

            mlp_prob = float(mlp_model.predict(x)[0][0])
            ensemble_pred = int(ensemble_model.predict(x)[0])

            return Response({
                "mlp_probability": mlp_prob,
                "ensemble_prediction": ensemble_pred
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=500
            )
