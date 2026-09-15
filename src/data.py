"""Funções de carregamento e pré-processamento de dados.

Isolar essa lógica aqui permite reutilizá-la tanto no treino
quanto na API de inferência, garantindo que os dados sejam
tratados exatamente da mesma forma nos dois contextos.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Churn"


def load_raw_data(path: str) -> pd.DataFrame:
    """Carrega o CSV bruto do disco."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza básica: trata TotalCharges e remove nulos.

    TotalCharges vem como string no dataset original por causa de
    alguns registros com espaço em branco em vez de número.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


def split_features_target(df: pd.DataFrame):
    """Separa features numéricas do target.

    Por enquanto só features numéricas (baseline). Colunas
    categóricas entram em uma iteração futura do pipeline.
    """
    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Executa o split treino/teste e escala as features.

    Retorna também o scaler ajustado, pois ele precisa ser salvo
    junto com o modelo — a API vai precisar dele para transformar
    novas requisições da mesma forma.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler