# TODOs

## Phase I:

1. General fixes
- [x] fix code so that training works
- [x] stores weights in ./artifacts/<date>.csv file
- [x] tests: that SGD converges properly on known data
- [x] fix dependencies
- [x] __init__ is missing docstring
- [x] some docstrings aren't correct; generate_data() says it's from a normal distrib
- [x] signatures missing everywhere
2. dockerized operation for **training** 
- [x] training is executable as a docker container
3. model serving
- [x] create flask app
- [x] load csv/pickle, serve predictions /stream, /batch
- [x] check model predictions work for /stream, document how to make a POST request in README
- [x] make flask app runnable as a docker container
4. CI
- [x] make project pip installable
- [x] add github actions CI pipeline [run tests; build image]
- [ ] add deployment instructions (just pull image)
 
## Phase II: MLFlow

- [x] training stores params, scores and artifacts (train/test sets, and plots) using mlflow
- [x] store model and create simple logic to upgrade it to production on training success
- [x] flask model loads current production model if mlflow server reachable; some local weights if not
- [x] add mlflow model server to docker-compose
- [x] TOO SLOW, leave out; track training losses

