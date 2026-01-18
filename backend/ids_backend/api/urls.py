from django.urls import path,include
from .views import UploadPCAPAPIView, PredictAPIView, PredictFromCSVAPIView

urlpatterns = [
    path("upload-pcap/", UploadPCAPAPIView.as_view()),
    path("predict/", PredictAPIView.as_view(), name="predict"),
    path("predict-csv/", PredictFromCSVAPIView.as_view(), name="predict-csv"),
]
