# Endeavour: ML End-to-End



### ToDos

* phase I

1. fix code so that training works
  . stores weights in ./artifacts/<date>.csv file (or pickle?)
  . tests: that SGD converges properly on known data
  . (fix dependencies and other code clean ups; __init__ is missing docstring; seems some docstrings arent correct, generate_data() says its from a normal distrib? signatures missing everywhere)
2. fix so training is executable as a docker container
3. add github actions CI pipeline [run tests; build image]
 
4. create small flask app
   . load csv/pickle, serve predictions /stream, /batch
5. make flask app runnable as a docker container
6. create docker-compose file to run the flask app


* phase II: mlflow

1. change so that training stores params, scores and artifacts (train/test sets, and plots) using mlflow (does not need mlflow server; just locally)
   . should be able to remove all print statements
2. store model and create simple logic to upgrade it to production on training success
3. setup so mlflow serve runs whatever is the production model
4. add mlflow model server to docker-compose
5. flask calls /invocations from mlflow serve