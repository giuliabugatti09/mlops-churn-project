# ---------- Estágio 1: build ----------
# Usamos uma imagem completa aqui só para instalar as dependências,
# incluindo as que exigem compilação. Essa camada é descartada depois.
FROM python:3.12-slim AS builder

WORKDIR /app

# Copiamos só o requirements.txt primeiro — isso aproveita o cache
# do Docker: se o código mudar mas as dependências não, essa camada
# não é reconstruída, economizando tempo de build.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ---------- Estágio 2: runtime ----------
# Imagem final, enxuta — só o necessário para rodar a API.
FROM python:3.12-slim

WORKDIR /app

# Copia os pacotes já instalados do estágio de build,
# sem trazer ferramentas de compilação junto.
COPY --from=builder /root/.local /root/.local

# Garante que os scripts instalados via --user sejam encontrados
ENV PATH=/root/.local/bin:$PATH

# Copia o código da aplicação e os artefatos do modelo
COPY app/ ./app/
COPY src/ ./src/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

# Porta que a API vai expor
EXPOSE 8000

# Sem --reload aqui — isso é só para desenvolvimento local
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]