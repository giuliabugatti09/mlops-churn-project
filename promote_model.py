"""Script pontual para promover uma versão do modelo registrado
para o estágio 'Staging' no MLflow Model Registry.
"""
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

client.transition_model_version_stage(
    name="churn-classifier",
    version=1,
    stage="Staging",
)

print("Versao 1 promovida para Staging")