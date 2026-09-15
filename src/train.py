import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Aponta o MLflow para o mesmo backend que subimos no terminal
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn-baseline")

# Carrega os dados (ajuste o nome do arquivo conforme o seu CSV)
df = pd.read_csv("data/raw/telco_churn.csv")

# Pré-processamento mínimo só para o baseline rodar
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run(run_name="logistic-regression-baseline"):
    max_iter = 200
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Loga os hiperparâmetros
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)

    # Loga as métricas
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

    # Loga o modelo como artefato
    mlflow.sklearn.log_model(model, "model")

    print("Run finalizado. Confira em http://127.0.0.1:5000")