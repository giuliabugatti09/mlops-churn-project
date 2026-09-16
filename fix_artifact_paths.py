"""Corrige os caminhos de artefato gravados no MLflow, trocando
o caminho absoluto do Windows por um caminho relativo — necessário
para o banco funcionar tanto localmente quanto dentro do container.
"""
import sqlite3

OLD_PREFIX = "file:C:/Users/gfbugatti/mlops-churn-project/"
NEW_PREFIX = "file:"

conn = sqlite3.connect("mlflow.db")
cursor = conn.cursor()

cursor.execute(
    "UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, ?)",
    (OLD_PREFIX, NEW_PREFIX),
)
print(f"Experiments atualizados: {cursor.rowcount}")

cursor.execute(
    "UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?)",
    (OLD_PREFIX, NEW_PREFIX),
)
print(f"Runs atualizados: {cursor.rowcount}")

conn.commit()
conn.close()