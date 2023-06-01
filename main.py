import numpy as np
from flask import Flask, request

from ml_e2e.simple_linear_regr import SimpleLinearRegression

app = Flask(__name__)


@app.route("/stream", methods=["POST"])
def stream():
    model = SimpleLinearRegression()
    model.load_weights()

    x_req = request.form.get('x')
    X_test = np.array([[x_req]])
    y_test_hat = model.predict(X_test)

    return {"result": y_test_hat}, 200


@app.route("/health")
def health_check():
    return "Ok", 200


if __name__ == "__main__":
    app.run()
