"""Funções de avaliação de modelos.

Centralizar as métricas aqui garante que treino e futuras
comparações usem sempre o mesmo critério de avaliação.
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcula o conjunto padrão de métricas para o projeto."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }