"""Arquiva a versão antiga do modelo e promove a versão correta
(com Pipeline completo) para Production.
"""
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

# Versão 1 é o modelo antigo, sem o scaler embutido no Pipeline —
# arquivamos para deixar claro que não é mais candidata a uso.
client.transition_model_version_stage(
    name="churn-classifier",
    version=1,
    stage="Archived",
)
print("Versao 1 arquivada")

# Versão 2 é o Pipeline completo (scaler + classificador), já
# validado pela API e pelos testes automatizados.
client.transition_model_version_stage(
    name="churn-classifier",
    version=2,
    stage="Production",
)
print("Versao 2 promovida para Production")