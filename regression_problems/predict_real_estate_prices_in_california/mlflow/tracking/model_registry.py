import mlflow
import yaml

def load_config():
    with open("./mlflow/config/mlflow_config.yaml", "r") as file:
        return yaml.safe_load(file)

def register_model(run_id, model_name):
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Registrar o modelo no MLflow
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"Modelo registrado com sucesso: {mv.name}, versão {mv.version}")
