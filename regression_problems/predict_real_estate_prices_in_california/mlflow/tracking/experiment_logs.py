import mlflow
import mlflow.sklearn
import yaml

# Carregar configurações do MLflow
def load_config():
    with open("./mlflow/config/mlflow_config.yaml", "r") as file:
        return yaml.safe_load(file)

def log_experiment(parameters, metrics, model):
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Logar parâmetros, métricas e modelo
        for param, value in parameters.items():
            mlflow.log_param(param, value)
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        mlflow.sklearn.log_model(model, "model")
        print("Experimento registrado no MLflow!")
