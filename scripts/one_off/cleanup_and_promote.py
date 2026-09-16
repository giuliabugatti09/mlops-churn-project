"""Arquiva a versão anterior e promove a nova versão (com
FEATURE_COLUMNS centralizado) para Production.
"""
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

# Versão 2 era o Pipeline anterior, sem a ordem de features centralizada.
client.transition_model_version_stage(
    name="churn-classifier",
    version=2,
    stage="Archived",
)
print("Versao 2 arquivada")

# Versão 3 é a atual, com FEATURE_COLUMNS como fonte única da verdade.
client.transition_model_version_stage(
    name="churn-classifier",
    version=3,
    stage="Production",
)
print("Versao 3 promovida para Production")