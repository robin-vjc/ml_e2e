import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from ml_e2e.simple_linear_regr import SimpleLinearRegression
from ml_e2e.utils import evaluate, generate_data


def test_sgd_matches_pseudo_inverse_solution():
    """
    Since the dataset used in these tests fits in memory, we can find the optimal fit directly by
    computing the pseudo inverse.
    """
    X_train, y_train, _, _ = generate_data()
    A = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    W_star, b_star = np.linalg.lstsq(A, y_train, rcond=None)[0]

    model = SimpleLinearRegression(iterations=15_000)
    model.fit(X_train, y_train)

    assert abs(W_star - model.W) <= 10
    assert abs(b_star - model.b) <= 2


def test_sgd_parameter_convergence_on_known_dataset():
    X_train, y_train, X_test, y_test = generate_data()
    model = SimpleLinearRegression(iterations=15_000)
    model.fit(X_train, y_train)

    assert 920 <= model.W <= 940
    assert 151 <= model.b <= 153

    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)


def test_model_performance_on_known_dataset():
    X_train, y_train, X_test, y_test = generate_data()
    model = SimpleLinearRegression(iterations=15_000)
    model.fit(X_train, y_train)

    y_test_hat = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_hat)
    r2 = r2_score(y_test, y_test_hat)

    assert r2 >= 0.47
    assert mse <= 2565
