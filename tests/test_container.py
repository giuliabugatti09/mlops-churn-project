"""Testes de integração que validam a imagem Docker de ponta a ponta:
build, execução do container e chamadas HTTP reais contra a API
rodando dentro dele. Diferente de tests/test_api.py, que testa o
código diretamente via TestClient, este arquivo valida o artefato
de deploy real.
"""
import subprocess
import time
import requests
import pytest

IMAGE_NAME = "churn-api:test"
CONTAINER_NAME = "churn-api-test-container"
PORT = 8001  # porta diferente da 8000, para não conflitar com um container já rodando localmente
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def running_container():
    """Builda a imagem, sobe o container, espera ficar pronto,
    executa os testes, e garante limpeza no final (mesmo se um
    teste falhar).
    """
    subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        check=True,
    )

    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", CONTAINER_NAME,
         "-p", f"{PORT}:8000", IMAGE_NAME],
        check=True,
    )

    # Espera a API ficar pronta, tentando /health por até 30 segundos
    for _ in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        subprocess.run(["docker", "stop", CONTAINER_NAME])
        pytest.fail("Container não ficou pronto a tempo")

    yield  # os testes rodam aqui

    # Limpeza garantida, mesmo se algum teste falhar
    subprocess.run(["docker", "stop", CONTAINER_NAME], check=False)


def test_container_health(running_container):
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_container_model_info(running_container):
    response = requests.get(f"{BASE_URL}/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "churn-classifier"
    assert body["stage"] == "Production"


def test_container_predict(running_container):
    payload = {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.0,
        "SeniorCitizen": 0,
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in [0, 1]
    assert 0.0 <= body["churn_probability"] <= 1.0