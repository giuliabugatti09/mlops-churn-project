"""Inspeciona os caminhos de artefato gravados no MLflow em todas
as tabelas relevantes: experiments, runs e model_versions.
"""
import sqlite3

conn = sqlite3.connect("mlflow.db")
cursor = conn.cursor()

print("=== model_versions (name, version, source) ===")
cursor.execute("SELECT name, version, source FROM model_versions")
for row in cursor.fetchall():
    print(row)

conn.close()