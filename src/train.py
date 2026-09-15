"""Orquestra o pipeline de treino: carrega dados, treina o modelo
e loga tudo no MLflow. Não implementa lógica de pré-processamento
ou de métricas diretamente — isso vive em data.py e evaluate.py.
"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from src.data import load_raw_data, clean_data, prepare_train_test
from src.evaluate import compute_metrics

DATA_PATH = "data/raw/telco_churn.csv"
EXPERIMENT_NAME = "churn-baseline"


def run_training(max_iter: int = 1000, run_name: str = "logistic-regression"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(df)

    with mlflow.start_run(run_name=run_name):
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"Run finalizado — métricas: {metrics}")


if __name__ == "__main__":
    run_training()