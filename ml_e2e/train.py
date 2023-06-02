import os
from typing import Dict

import mlflow

from ml_e2e.simple_linear_regr import SimpleLinearRegression, SLRWrapper
from ml_e2e.utils import generate_data, get_scores


def train_model():
    X_train, y_train, X_test, y_test = generate_data()

    mlflow_host = os.getenv("MLFLOW_SERVER", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_host)

    with mlflow.start_run() as run:
        iterations = 15000
        lr = 0.1
        model = SimpleLinearRegression(iterations=iterations, lr=lr)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        scores = get_scores(y_test, y_pred)

        mlflow.log_param("iterations", iterations)
        mlflow.log_param("lr", lr)
        mlflow.log_metrics(scores)

        if performance_check(scores):
            model_path = f"slr-{run.info.run_uuid}"
            reg_model_name = "SimpleLinearRegression"
            mlflow_model = SLRWrapper(model=model)

            # save artifacts
            mlflow.pyfunc.save_model(
                path=f"../artifacts/{run.info.experiment_id}/{run.info.run_uuid}/artifacts/{model_path}",
                python_model=mlflow_model,
            )

            # log model
            mlflow.pyfunc.log_model(
                artifact_path=model_path,
                python_model=mlflow_model,
                registered_model_name=reg_model_name,
            )

            # get registered version and promote to prod
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(reg_model_name)[0].version
            client.transition_model_version_stage(
                name=reg_model_name,
                version=latest_version,
                stage="Production"
            )


def performance_check(scores: Dict[str, float]) -> bool:
    if scores["r2"] >= 0.4:
        return True

    return False


if __name__ == "__main__":
    train_model()