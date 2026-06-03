from functools import lru_cache
from pathlib import Path
from typing import Any
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src.api.pydantic_models import PredictionRequest, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    (PROJECT_ROOT / "mlruns").as_uri(),
)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CreditRiskModel")


app = FastAPI(
    title="Credit Risk Probability API",
    version="1.0.0",
    description=(
        "FastAPI service that scores credit-risk probability with MLflow."
    ),
)


def configure_tracking() -> None:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)


@lru_cache(maxsize=1)
def resolve_model_uri() -> str:
    explicit_model_uri = os.getenv("MLFLOW_MODEL_URI")
    if explicit_model_uri:
        return explicit_model_uri

    configure_tracking()
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        raise RuntimeError(
            f"No registered versions found for model '{MODEL_NAME}'."
        )

    latest_version = max(
        versions,
        key=lambda model_version: int(model_version.version),
    )
    return f"models:/{MODEL_NAME}/{latest_version.version}"


@lru_cache(maxsize=1)
def load_model() -> Any:
    configure_tracking()
    model_uri = resolve_model_uri()
    return mlflow.sklearn.load_model(model_uri)


def prepare_features(features: dict[str, Any], model: Any) -> pd.DataFrame:
    frame = pd.DataFrame([features])
    expected_columns = list(getattr(model, "feature_names_in_", frame.columns))
    aligned_frame = frame.reindex(columns=expected_columns, fill_value=0)

    for column_name in aligned_frame.columns:
        aligned_frame[column_name] = pd.to_numeric(
            aligned_frame[column_name],
            errors="ignore",
        )

    return aligned_frame


def predict_probability(model: Any, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", []))

        if 1 in classes:
            return float(probabilities[classes.index(1)])

        if len(probabilities) > 1:
            return float(probabilities[-1])

        return float(probabilities[0])

    prediction = model.predict(features)[0]
    return float(prediction)


@app.on_event("startup")
def startup_event() -> None:
    try:
        app.state.model_uri = resolve_model_uri()
        app.state.model = load_model()
        app.state.model_error = None
    except Exception as exc:  # pragma: no cover - startup fallback
        app.state.model_uri = None
        app.state.model = None
        app.state.model_error = str(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if app.state.model is not None else "degraded",
        "model_uri": app.state.model_uri,
        "model_error": app.state.model_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.model_error or "Model is not available.",
        )

    feature_frame = prepare_features(request.features, app.state.model)
    risk_probability = predict_probability(app.state.model, feature_frame)
    predicted_class = int(risk_probability >= 0.5)

    return PredictionResponse(
        risk_probability=risk_probability,
        predicted_class=predicted_class,
        model_uri=app.state.model_uri,
    )
