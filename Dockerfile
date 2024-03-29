FROM python:3.11 AS base

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY requirements/ .
RUN pip install --no-cache-dir -r base.txt


FROM base AS test

RUN pip install --no-cache-dir -r test.txt
COPY . .

CMD ["pytest", "--isort", "--black", "--pylint", "--cov", "."]


FROM base AS web

EXPOSE 8000
COPY . .

CMD ["flask", "--app", "main", "run", "--host=0.0.0.0", "--port=8000", "--debug"]


FROM base AS mlflow

EXPOSE 5000
COPY . .

CMD ["mlflow", "server", "--backend-store-uri=sqlite:///mlflow.db", "--default-artifact-root=./artifacts", "--host=0.0.0.0", "--port=5000"]
