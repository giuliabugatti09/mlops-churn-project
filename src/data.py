"""Funções de carregamento e pré-processamento de dados.

Isolar essa lógica aqui permite reutilizá-la tanto no treino
quanto na API de inferência, garantindo que os dados sejam
tratados exatamente da mesma forma nos dois contextos.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
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
    """Executa o split treino/teste. A escala das features agora
    faz parte do Pipeline do modelo (ver train.py), não deste módulo —
    isso evita divergência entre o pré-processamento do treino e o
    da API de inferência.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
