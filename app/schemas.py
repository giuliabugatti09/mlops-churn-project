"""Contratos de entrada e saída da API, validados automaticamente
pelo FastAPI via Pydantic.
"""
from pydantic import BaseModel, Field


class ChurnFeatures(BaseModel):
    """Features numéricas esperadas para uma predição de churn.

    Os nomes dos campos espelham as colunas numéricas do dataset
    original usadas no treino (ver src/data.py).
    """
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    MonthlyCharges: float = Field(..., ge=0, description="Cobrança mensal")
    TotalCharges: float = Field(..., ge=0, description="Total cobrado até hoje")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 se for idoso, 0 caso contrário")

    class Config:
        json_schema_extra = {
            "example": {
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 70.5,
                "TotalCharges": 845.0,
            }
        }


class PredictionResponse(BaseModel):
    """Resposta da predição de churn."""
    churn_prediction: int = Field(..., description="1 = vai cancelar, 0 = não vai cancelar")
    churn_probability: float = Field(..., description="Probabilidade de churn, entre 0 e 1")