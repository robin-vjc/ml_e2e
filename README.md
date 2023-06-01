# Endeavour: ML End-to-End

## System Design

* Training observability is ensured through experiment tracking using MLFlow
* Training can be triggered either manually (developers running code locally) or by CI pipeline
* The model is served by a custom flask server instead of relying on `mlflow serve`, to ensure that
  1. appropriate security could easily be added to the project if this service was to face the public internet
  2. we can adhere to the required API specification (`/stream` and `/batch`)
* Notable simplifications
  * No load balancer when serving the model
  * No k8s deployment. We push images to github's registry in CI though, which facilitates deployment to k8s
  * No model staging; we would usually set a model to staging, test its performance by serving a small fraction of clients (1%) using a load balancer, which we'd rgadually increase (5%, 10% etc). Once we are sure there are no problems (performance regressions, crashes, systems overloads etc..) we'd promote to prod.
  * Artifacts are stored on the local FS. We'd usually set up a remote MLFlow server with artifacts stored in s3


### Training Pipeline
![training_pipeline.png](docs/img/training_pipeline.png)

### Inference
![inference.png](docs/img/inference.png)



### ToDos

Phase I:

1. General fixes
- [x] fix code so that training works
- [ ] stores weights in ./artifacts/<date>.csv file (or pickle?)
- [x] tests: that SGD converges properly on known data
- [ ] fix dependencies and other code clean ups
- [x] __init__ is missing docstring
- [ ] some docstrings aren't correct; generate_data() says it's from a normal distrib
- [ ] signatures missing everywhere
2. dockerized operation for **training** 
- [ ] training is executable as a docker container
3. model serving
- [ ] create flask app
- [ ] load csv/pickle, serve predictions /stream, /batch
- [ ] make flask app runnable as a docker container
4. CI
- [ ] add github actions CI pipeline [run tests; build image]
 

Phase II: MLFlow

- [ ] training stores params, scores and artifacts (train/test sets, and plots) using mlflow
   * should be able to remove all print statements
- [ ] store model and create simple logic to upgrade it to production on training success
- [ ] flask model loads current production model if ml server reachable; some local weights if not
- [ ] add mlflow model server to docker-compose
