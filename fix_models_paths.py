import sqlite3
conn = sqlite3.connect("mlflow.db")
cursor = conn.cursor()
cursor.execute("SELECT name, version, source FROM model_versions")
for row in cursor.fetchall():
    print(row)
conn.close()