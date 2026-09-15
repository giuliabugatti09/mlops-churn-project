"""Script pontual para inspecionar a ordem de features que o
Pipeline em produção espera, direto do modelo treinado.
"""
from app.model_loader import load_model

model = load_model()
print("Ordem de features esperada pelo modelo:", list(model.feature_names_in_))