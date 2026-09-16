"""Script pontual para registrar uma versão específica do modelo
no MLflow Model Registry, a partir de um run já existente.
"""
import mlflow

RUN_ID = "80a3550b39654e8fb4825a685140f411"
MODEL_NAME = "churn-classifier"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = f"runs:/{RUN_ID}/model"
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

print(f"Modelo registrado: {result.name}, versão {result.version}")

from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

client.transition_model_version_stage(
    name="churn-classifier",
    version=result.version,
    stage="Staging",
)

print(f"Versão {result.version} promovida para Staging")