"""Lógica de carregamento do modelo a partir do MLflow Model Registry.

Centralizar isso aqui permite trocar a fonte do modelo (ex: de
Staging para Production) sem tocar em main.py.
"""
import mlflow.sklearn
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-classifier"
MODEL_STAGE = "Production"


def load_model():
    """Carrega o pipeline (scaler + classificador) do Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.sklearn.load_model(model_uri)


def get_model_info() -> dict:
    """Retorna metadados da versão do modelo atualmente em uso.

    Consultar o Registry aqui (em vez de fixar a versão como
    constante) garante que /model-info sempre reflita a verdade,
    mesmo que o modelo em Staging seja trocado depois do deploy.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        return {"name": MODEL_NAME, "stage": MODEL_STAGE, "version": None}

    latest = versions[0]
    return {
        "name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "version": latest.version,
        "run_id": latest.run_id,
    }
