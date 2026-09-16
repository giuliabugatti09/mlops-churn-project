"""Corrige o caminho de storage_location na tabela model_versions —
essa coluna é priorizada por get_model_version_download_uri() e
havia ficado com o caminho absoluto do Windows mesmo depois da
correção da coluna 'source'.
"""
import sqlite3

OLD_PREFIX = "file:C:/Users/gfbugatti/mlops-churn-project/"
NEW_PREFIX = "file:"

conn = sqlite3.connect("mlflow.db")
cursor = conn.cursor()

cursor.execute(
    "UPDATE model_versions SET storage_location = REPLACE(storage_location, ?, ?)",
    (OLD_PREFIX, NEW_PREFIX),
)
print(f"storage_location atualizado: {cursor.rowcount}")

conn.commit()
conn.close()