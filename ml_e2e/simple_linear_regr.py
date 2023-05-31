import numpy as np
from sklearn.metrics import mean_squared_error

from ml_e2e.utils import generate_data, evaluate


class SimpleLinearRegression:
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
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def __sgd(self, X: np.array, y: np.array, y_hat: np.array) -> None:
        """
        Training loop, uses SGD.
        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        """
        n = X.shape[0]

        dW = 2/n * np.sum(X*(y_hat - y), axis=0)
        db = 2/n * np.sum((y_hat - y), axis=0)
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
        y_hat = self.W*X + self.b
        return y_hat


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()
    model = SimpleLinearRegression()
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)
