# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

MODEL_PATH = "model.pkl"

app = FastAPI(
    title="End-to-End MLOps Iris Classifier",
    description="FastAPI service to serve ML model with basic monitoring",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total number of prediction requests"
)
REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds", "Latency of prediction requests"
)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def load_model():
    obj = joblib.load(MODEL_PATH)
    return obj["model"], obj["target_names"]

# load model at startup
model, target_names = load_model()

@app.get("/")
def root():
    return {"message": "Iris MLOps API is running"}

@app.post("/predict")
def predict(features: IrisFeatures):
    start_time = time.time()
    REQUEST_COUNT.inc()

    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    y_pred = model.predict(X)[0]
    class_name = target_names[y_pred]

    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)

    return {
        "predicted_class_id": int(y_pred),
        "predicted_class_name": class_name,
        "latency_seconds": latency
    }

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
