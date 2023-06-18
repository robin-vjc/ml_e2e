import json
import os
import time
from typing import Union

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error

from ml_e2e.utils import ARTIFACTS_PATH, evaluate, generate_data, get_scores


class LinearRegression:
    def __init__(self, iterations: int = 15000, lr: float = 0.1):
        """
        A linear regression model trained with SGD.
        :param iterations: number of iterations the fit method will be called
        :param lr: learning rate
        """
        self.iterations = iterations
        self.lr = lr
        self.losses = []  # A list to hold the history of the calculated losses
        self.W, self.b = None, None  # the slope and the intercept of the model

    def __loss(self, y: np.array, y_hat: np.array) -> float:
        """
        Mean squared error
        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: mean squared error
        """
        loss = mean_squared_error(y, y_hat)
        self.losses.append(loss)
        return loss

    def __init_weights(self, X: np.array) -> None:
        """
        :param X: The training set
        """
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[: X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def __sgd(self, X: np.array, y: np.array, y_hat: np.array) -> None:
        """
        Training loop, uses SGD.
        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        """
        n = X.shape[0]

        dW = 2 / n * np.sum(X * (y_hat - y), axis=0)
        db = 2 / n * np.sum((y_hat - y), axis=0)
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the least squares model
        :param X: The training set
        :param y: The true output of the training set
        """
        self.__init_weights(X)
        y_hat = self.predict(X)
        loss = self.__loss(y, y_hat)
        print(f"Initial Loss: {loss}")

        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X: np.array) -> np.array:
        """
        Returns model's predictions
        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        y_hat = self.W * X + self.b
        return y_hat

    def save_weights(self) -> None:
        """
        Stores the model weights in artifacts
        """
        unix_time = int(time.time())
        weights_file_path = ARTIFACTS_PATH / "weights" / f"{unix_time}.json"
        weights = {"W": self.W[0][0], "b": self.b[0]}

        with open(weights_file_path, "w", encoding="utf-8") as f:
            json.dump(weights, f)

    def load_weights(self) -> None:
        """
        Loads the latest weights found in artifacts
        """
        weight_files = os.listdir(ARTIFACTS_PATH / "weights")
        weight_files = sorted(weight_files, reverse=True)

        try:
            latest_weights = weight_files[0]
        except IndexError as e:
            raise IndexError(
                "Unable to locate stored model weights in artifacts. Have you run the training pipeline yet?"
            ) from e

        latest_weights_path = ARTIFACTS_PATH / "weights" / latest_weights
        with open(latest_weights_path, "r", encoding="utf-8") as f:
            weights = json.load(f)

        self.W = np.array([[weights["W"]]])
        self.b = np.array([weights["b"]])


class SLRWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper for the SimpleLinearModel that can be stored as a MLFlow model."""

    def __init__(self, slr_model: LinearRegression):
        self.model = slr_model

    def predict(self, context, model_input) -> np.array:
        return self.model.predict(model_input)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)
    scores = get_scores(y_test, predicted)

    if scores["r2"] >= 0.4:
        model.save_weights()


def load_model() -> Union[LinearRegression, SLRWrapper]:
    """
    Attempts to load the model from the MLServer, reverts to a local version if that fails.
    """
    model_served = os.getenv("MODEL_SERVED", "local")

    if model_served == "local":
        print("Loading SimpleLinearRegression model from local artifacts...")
        loaded_model = LinearRegression()
        loaded_model.load_weights()
    elif model_served == "mlflow":
        print("Loading SimpleLinearRegression model from MLFlow server...")
        mlflow_host = os.getenv("MLFLOW_SERVER_HOST", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_host)
        reg_model_name = "SimpleLinearRegression"
        model_uri = f"models:/{reg_model_name}/Production"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
    else:
        raise ValueError(
            "Unrecognized model to serve; should be either 'local' or 'mlflow'."
        )

    return loaded_model
