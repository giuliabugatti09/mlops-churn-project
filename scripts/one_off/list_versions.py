"""Script pontual para listar todas as versões registradas
de um modelo no MLflow Model Registry, com seu estágio atual.
"""
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

versions = client.search_model_versions("name='churn-classifier'")

for v in versions:
    print(f"Versao {v.version} - Stage: {v.current_stage} - Run: {v.run_id}")