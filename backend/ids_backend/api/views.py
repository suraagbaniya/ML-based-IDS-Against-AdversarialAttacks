import os
import joblib
import numpy as np
import pandas as pd

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.core.files.storage import default_storage

# ==============================
# LOAD MODELS (ONCE)
# ==============================

BASE_DIR = settings.BASE_DIR
# Actual locations in the workspace
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

ENSEMBLE_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "ids_model", "ensemble.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "scaler.pkl")

ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==============================
# HELPER FUNCTIONS
# ==============================

def prepare_features(features):
    """
    Convert raw features to model-ready format
    """
    x = np.array(features).reshape(1, -1)
    x = scaler.transform(x)
    return x


def load_features_from_csv(csv_path):
    """
    Load extracted features from NetworkMiner CSV
    (uses first row for prediction)
    """
    df = pd.read_csv(csv_path)
    features = df.iloc[0].values.tolist()
    return features


# ==============================
# API VIEWS
# ==============================

class PredictAPIView(APIView):
    """
    Predict intrusion from feature vector
    """

    def post(self, request):
        try:
            raw_features = request.data.get("features")

            if raw_features is None:
                return Response(
                    {"error": "No features provided"},
                    status=400
                )

            x = prepare_features(raw_features)

            prediction = int(ensemble.predict(x)[0])
            probability = float(ensemble.predict_proba(x)[0][1])

            return Response({
                "prediction": prediction,
                "label": "Attack" if prediction == 1 else "Benign",
                "attack_probability": round(probability, 4)
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=500
            )


class UploadPCAPAPIView(APIView):
    """
    Upload PCAP file (for NetworkMiner processing)
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        try:
            pcap_file = request.FILES.get("pcap")

            if not pcap_file:
                return Response(
                    {"error": "No PCAP file uploaded"},
                    status=400
                )

            save_path = default_storage.save(
                f"uploads/pcaps/{pcap_file.name}",
                pcap_file
            )

            return Response({
                "message": "PCAP uploaded successfully",
                "path": save_path
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=500
            )


class PredictFromCSVAPIView(APIView):
    """
    Predict intrusion directly from NetworkMiner CSV output
    """

    def post(self, request):
        try:
            csv_path = request.data.get("csv_path")

            if not csv_path:
                return Response(
                    {"error": "CSV path not provided"},
                    status=400
                )

            features = load_features_from_csv(csv_path)
            x = prepare_features(features)

            prediction = int(ensemble.predict(x)[0])
            probability = float(ensemble.predict_proba(x)[0][1])

            return Response({
                "prediction": prediction,
                "label": "Attack" if prediction == 1 else "Benign",
                "attack_probability": round(probability, 4)
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=500
            )
