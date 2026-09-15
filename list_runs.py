"""Lista todos os runs do experimento churn-baseline, do mais
recente para o mais antigo, para localizar o run correto a registrar.
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

runs = mlflow.search_runs(experiment_names=["churn-baseline"], order_by=["start_time DESC"])
print(runs[["run_id", "tags.mlflow.runName", "start_time", "metrics.test_roc_auc"]])