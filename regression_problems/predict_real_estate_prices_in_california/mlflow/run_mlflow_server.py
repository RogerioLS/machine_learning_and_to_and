import os
import subprocess

def start_mlflow_server():
    # Diretório onde os artefatos serão armazenados
    artifact_dir = os.path.abspath("./mlflow/mlruns")
    os.makedirs(artifact_dir, exist_ok=True)

    # Comando para iniciar o servidor MLflow
    command = [
        "mlflow",
        "server",
        "--backend-store-uri", artifact_dir,
        "--default-artifact-root", artifact_dir,
        "--host", "127.0.0.1",
        "--port", "5000"
    ]

    print("Iniciando servidor MLflow...")
    subprocess.run(command)

if __name__ == "__main__":
    start_mlflow_server()
