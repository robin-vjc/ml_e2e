import numpy as np
from flask import Flask, request

from ml_e2e.simple_linear_regr import load_model

app = Flask(__name__)
model = load_model()


@app.route("/stream", methods=["POST"])
def stream():
    X_req = request.json["X"]
    X_req = np.array([[X_req]])

    y_pred = model.predict(X_req)

    return {"result": y_pred[0][0]}, 200


@app.route("/batch", methods=["POST"])
def batch():
    X_req = request.json["X"]
    X_req = np.array([X_req])
    y_pred = model.predict(X_req)
    y_pred = y_pred[0].tolist()

    return {"result": y_pred}, 200


@app.route("/health")
def health_check():
    return "Ok", 200


if __name__ == "__main__":
    app.run()
