from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

client.transition_model_version_stage(
    name="churn-classifier",
    version=2,
    stage="Production",
)

print("Versao 2 promovida para Production")