"""Orquestra o pipeline de treino: carrega dados, treina modelos
candidatos (empacotados com pré-processamento) e loga tudo no MLflow.
"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_raw_data, clean_data, prepare_train_test
from src.evaluate import compute_metrics

DATA_PATH = "data/raw/telco_churn.csv"
EXPERIMENT_NAME = "churn-baseline"


def train_logistic_regression(X_train, y_train, X_test, y_test, max_iter: int = 1000):
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="logistic-regression-pipeline"):
        # O Pipeline garante que o mesmo scaler ajustado no treino
        # seja aplicado a qualquer dado novo que passar por .predict()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=max_iter, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        print(f"[LogisticRegression pipeline] métricas de teste: {metrics}")


def run_training():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    train_logistic_regression(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    run_training()