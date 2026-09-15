"""Aplicação FastAPI que serve o modelo de predição de churn."""
import pandas as pd
from fastapi import FastAPI

from app.schemas import ChurnFeatures, PredictionResponse
from app.model_loader import load_model, get_model_info

app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para o modelo de predição de churn de clientes.",
    version="1.0.0",
)

model = load_model()


@app.get("/health")
def health():
    """Usado por sistemas de orquestração para checar se a API está viva."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Expõe metadados do modelo atualmente carregado, para rastreabilidade."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ChurnFeatures):
    """Recebe as features de um cliente e retorna a predição de churn."""
    input_df = pd.DataFrame([features.model_dump()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return PredictionResponse(
        churn_prediction=int(prediction),
        churn_probability=float(probability),
    )