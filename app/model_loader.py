"""Lógica de carregamento do modelo a partir do MLflow Model Registry.

Centralizar isso aqui permite trocar a fonte do modelo (ex: de
Staging para Production) sem tocar em main.py.
"""
import mlflow.sklearn

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-classifier"
MODEL_STAGE = "Staging"


def load_model():
    """Carrega o pipeline (scaler + classificador) do Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.sklearn.load_model(model_uri)