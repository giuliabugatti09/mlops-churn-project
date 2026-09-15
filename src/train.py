"""Orquestra o pipeline de treino: carrega dados, treina modelos
candidatos e loga tudo no MLflow para comparação.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data import load_raw_data, clean_data, prepare_train_test
from src.evaluate import compute_metrics

DATA_PATH = "data/raw/telco_churn.csv"
EXPERIMENT_NAME = "churn-baseline"


def train_logistic_regression(X_train, y_train, X_test, y_test, max_iter: int = 1000):
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="logistic-regression"):
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # autolog já captura hiperparâmetros e métricas de treino;
        # logamos manualmente só as métricas de teste, que são o
        # que realmente importa para decidir qual modelo é melhor.
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        print(f"[LogisticRegression] métricas de teste: {metrics}")


def train_xgboost(X_train, y_train, X_test, y_test, n_estimators: int = 100, max_depth: int = 4):
    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name="xgboost"):
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        print(f"[XGBoost] métricas de teste: {metrics}")


def run_training():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(df)

    train_logistic_regression(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    run_training()