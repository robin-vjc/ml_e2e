# Blueprint for ML Projects

This blueprint for ML projects entails various pieces that are usually required:
* MLFlow experiment tracking
* MLFlow model tracking and staging
* CI pipeline

## System Design

* Model observability (training, lineage, deployment staging, ...) is ensured via tracking with MLFlow
* The Github CI/CD pipeline runs tests on PR/merge to master, and on success pushes the image of the inference server to Github's registry, see the [packages](https://github.com/robin-vjc/ml_e2e/pkgs/container/endeavour_e2e_ml)
* Training is triggered manually (developers running code locally), and could be added to CI
  * if training computations could not fit/would take too long on a laptop, and had to be run on k8s (for instance if we had to scan a large grid of hyperparameters), then we'd add a cli.py that accepts such hyperparams and we could trigger the necessary runs in parallel
* Inference server has to modes: 1) use locally stored weights, or 2) fetch model weights from a mlflow server
* The model is served by a custom flask server instead of relying on `mlflow serve`, to ensure that
  1. appropriate security could easily be added to the project if this service was to face the public internet
  2. we can adhere to the required API specification (`/stream` and `/batch`)
* Notable simplifications
  * No load balancer when serving the model
  * No k8s deployment. We push images to Github's registry in CI though, which facilitates deployment to k8s
  * No model staging (we promote directly to prod if performance checks pass); we would usually set a model to staging, test its performance by serving a small fraction of clients (1%) using a load balancer, which we'd gradually increase (5%, 10% etc). Once we are sure there are no problems (performance regressions, crashes, systems overloads etc..) we'd promote to prod.
  * Artifacts are stored on the local FS. We'd usually set up a remote MLFlow server with artifacts stored in s3


### Training Pipeline
![training_pipeline.png](docs/img/training_pipeline.png)

### Inference
![inference.png](docs/img/inference.png)


## Running for local dev (docker)

Build the project images:
```bash
docker-compose build
```

Run the tests (includes detection of linting issues)
```bash
docker-compose run test
```

### Run inference
Start the stack (flask server has autoreload)
```bash
docker-compose up web -d
```
You can verify that the stack is running by checking the status of the flask server on `http://localhost:8000/health`, 
while the mlflow server is on `http://localhost:5000/`.

You can also verify that the stack is working correctly by requesting a prediction
```python
import requests

# test /stream
X_request = {'X': 0.2}
r = requests.post("http://localhost:8000/stream", json=X_request)
print(r.json())

# test /batch
X_request = {'X': [0.2, 0.5, -0.1]}
r = requests.post("http://localhost:8000/batch", json=X_request)
print(r.json())
```

### Run training
While the stack is running (we need MLFlow up to log models and metrics), training can be run as follows
```bash
# run stack if not already running
docker-compose up web -d

docker-compose run train
```
A new model will be trained and if performance checks pass, the model will be promoted to production:

![mlflow_ui.png](docs/img/mlflow_ui.png)

Model Registry:

![mlflow_model_registry.png](docs/img/mlflow_model_registry.png)


## Running for local dev (no docker)

It is recommended to work with the docker setup above. However, it is possible to install this project as an editable 
package for quick local iterations:
```bash
pip install -e .
```

## Deployment

A production image is created as part of CI/CD. We have 2 deployment options.

**Option 1**: run using locally stored weights:
```bash
docker run -e MODEL_SERVED=local -p 8000:8000 --rm ghcr.io/robin-vjc/endeavour_e2e_ml:latest
```

**Option 2**: a running MLFlow server is available (adjust hostname in the command accordingly) and the most recent production model can be loaded from it:
```bash
docker run -e MODEL_SERVED=mlflow -e MLFLOW_SERVER_HOST=http://mlflow:5000 --network=ml_e2e_ml_e2e -p 8000:8000 --rm ghcr.io/robin-vjc/endeavour_e2e_ml:latest
```
