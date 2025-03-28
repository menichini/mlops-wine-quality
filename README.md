# 🧪 MLOps com Wine Quality Dataset

Este projeto é uma demonstração prática de um pipeline de Machine Learning aplicado à regressão, utilizando o dataset Wine Quality (UCI), com práticas de MLOps: versionamento, tracking e deploy.

## 🎯 Objetivo

Prever a qualidade sensorial de vinhos com base em características físico-químicas, utilizando modelos de regressão e ferramentas open source para automação do ciclo de vida do modelo.

## 🚀 Tecnologias Utilizadas

- Python 3.10+
- Scikit-learn
- MLflow
- FastAPI
- Docker
- Kubernetes - Minikube
- JupyterLab

## 🧱 Estrutura do Projeto

- `data/`: Base de dados utilizada (Wine Quality - Red)
- `src/`: Scripts de treinamento, avaliação e salvamento do modelo
- `api/`: Serviço RESTful com FastAPI para consumo do modelo treinado
- `docker/`: Dockerfile da aplicação
- `kubernetes/`: YAMLs de deploy para Kubernetes
- `mlruns/`: Tracking local com MLflow
- `notebooks/`: Desenvolvimento exploratório com Jupyter

## ⚙️ Como Executar Localmente

1. Clone o repositório:

```bash
git clone https://github.com/menichini/mlops-wine-quality.git
cd mlops-wine-quality
```

2. Instale os pacotes:

```bash
pip install -r requirements.txt
```

3. Rode o treinamento:

```bash
python src/train.py
```

4. Sirva a API:

```bash
uvicorn api.main:app --reload
```

📦 Deploy com Docker

```bash
docker build -t wine-api:latest -f docker/Dockerfile .
docker run -p 8000:8000 wine-api:latest
```

☸️ Deploy no Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

📊 Exemplo de Previsão via API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}'
```
