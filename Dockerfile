FROM python:3.12-slim

WORKDIR /opt/ml

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow_model/ /opt/ml/model/

EXPOSE 8080

CMD ["mlflow", "models", "serve", \
     "-m", "/opt/ml/model", \
     "-h", "0.0.0.0", \
     "-p", "8080", \
     "--env-manager=local"]
