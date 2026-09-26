# Premier League Player Market Value Predictor (MLOps)

[![Continuous Integration](https://github.com/Adityajain8595/PL-Transfer-Market-Value-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Adityajain8595/PL-Transfer-Market-Value-Predictor/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![DagsHub / MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow)](https://dagshub.com)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6.svg?logo=dvc)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)

An end-to-end, production-grade MLOps system that models, tracks, evaluates, and serves market valuation predictions for Premier League football players using historical match appearances, performance trajectories, European competition experience, and demographic features.

---

## Architecture Overview

```
                      +-----------------------------------+
                      |   Kaggle API Ingestion Pipeline   |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------+-----------------+
                      |    DuckDB Feature Engineering     |
                      |  (Window Aggregations & Parquet)  |
                      +--------+-----------------+--------+
                               |                 |
                               v                 v
                 +-------------+----+      +-----+-------------+
                 | DVC Remote Sync  |      | Evidently Drift   |
                 | (DagsHub Storage)|      | Monitoring Suite  |
                 +------------------+      +-------------------+
                               |
                               v
                      +--------+--------------------------+
                      |   Model Tournament Arena (MLflow) |
                      | (LGBM, XGBoost, CatBoost, RF, GB) |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------+-----------------+
                      | Champion Gate & Registry Promotion|
                      +-----------------+-----------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
+----------------+----------------+          +-----------------+---------------+
|   FastAPI REST Inference API    | <------> |     Streamlit Analytics UI      |
|  (CORS, Prometheus, Batch, Sim) |          |  (Live Directory, What-If Sims) |
+----------------+----------------+          +---------------------------------+
                 |
                 v
+----------------+----------------+
|  Prometheus & Grafana Dashboard |
+---------------------------------+
```

---

## Key System Features

1. **Analytical Ingestion & Feature Engineering with DuckDB**:
   - Ingests raw match, player, valuation, and club datasets from Kaggle.
   - Executes zero-copy SQL window aggregations via DuckDB to calculate exact temporal intervals between valuation dates, match minutes, goals/assists per 90, European match volume, and disciplinary records without lookahead leakage.
   - Materializes partitioned, compressed Parquet datasets (`features.parquet` and `player_latest_stats.parquet`).

2. **Continuous Versioning with DVC**:
   - Large raw CSVs and processed Parquet files are tracked using Data Version Control (DVC) backed by DagsHub remote storage, keeping the Git repository lightweight.

3. **Multi-Model Tournament Arena & Registry Promotion**:
   - Evaluates 5 competing regressor architectures on strict chronological splits: **LightGBM**, **XGBoost**, **CatBoost**, **Random Forest**, and **Gradient Boosting**.
   - Logs metrics (MAE, RMSE, $R^2$), hyperparameters, and artifacts to MLflow / DagsHub.
   - Deploys an automated **Champion vs. Challenger quality gate**: candidates are only promoted if they strictly satisfy the regression performance threshold (&le; 5% degradation against the incumbent).

4. **Statistical Drift Monitoring**:
   - Automated distribution auditing via **Evidently AI** using Wasserstein distance and Population Stability Index (PSI) to detect feature drift and prediction skew across data ingestion cycles.

5. **Production Serving Layer (FastAPI)**:
   - High-throughput asynchronous REST API exposing single prediction (`/predict`), batch inference (`/predict/batch`), hydrated player lookup (`/predict/player/{id}`), and what-if transfer simulation (`/predict/simulate`).
   - Integrated `CORSMiddleware` and Prometheus metrics exporter (`/metrics`).

6. **Executive Dashboard (Streamlit)**:
   - Clean, modern design featuring Premier League branding.
   - Active player directory table, historical career valuation trajectories, and real-time "What-If" scenario simulator.

---

## Repository Structure

```
├── .github/workflows/       # GitHub Actions CI/CD workflows
├── configs/                 # Declarative YAML configurations (data, model, drift)
├── dags/                    # Apache Airflow orchestration DAGs
├── data/
│   ├── raw/                 # Raw Kaggle CSV files (DVC tracked)
│   └── processed/           # Engineered Parquet features (DVC tracked)
├── docker/
│   ├── Dockerfile.api       # Multi-stage container for FastAPI serving
│   ├── Dockerfile.frontend  # Container for Streamlit web app
│   ├── docker-compose.yml   # Multi-service stack (API, UI, Prometheus, Grafana)
│   ├── prometheus.yml       # Metrics scraping configuration
│   └── grafana/             # Pre-configured provisioning and dashboards
├── models/                  # Local champion model binary and tournament records
├── reports/                 # Drift audit summaries and reports
├── src/
│   ├── data/                # Ingestion and DVC synchronization
│   ├── features/            # DuckDB SQL feature engineering pipeline
│   ├── models/              # Tournament training, evaluation, and registration
│   ├── monitoring/          # Evidently statistical drift auditor
│   ├── serve/               # FastAPI application, schemas, and Streamlit frontend
│   └── utils/               # Structured logging, custom exceptions, paths
└── tests/                   # Pytest automated test suite
```

---

## Quickstart Guide

### 1. Prerequisites & Virtual Environment

- Python 3.10 or 3.11 recommended.
- Git & DVC installed.

```bash
# Clone the repository
git clone https://github.com/Adityajain8595/PL-Transfer-Market-Value-Predictor.git
cd PL-Transfer-Market-Value-Predictor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and provide your credentials:

```bash
cp .env.example .env
```

```env
# DagsHub / MLflow Credentials
DAGSHUB_USER_TOKEN=your_token
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# Kaggle API Credentials
KAGGLE_API_KEY=KGAT_your_api_token
```

### 3. Pipeline Execution

```bash
# 1. Download raw data from Kaggle
python src/data/download.py

# 2. Run DuckDB feature transformation
python src/features/transform.py

# 3. Train model tournament and promote champion
python src/models/train.py

# 4. Audit data drift against baseline
python src/monitoring/drift_detector.py
```

---

## Serving & Visualization

### Launch Locally

In two separate terminals:

```bash
# Terminal 1: Start FastAPI Service
uvicorn src.serve.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Streamlit Dashboard
streamlit run src/serve/frontend.py --server.port 8501
```

- **Streamlit Web Application**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Prometheus Metrics Endpoint**: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

---

## Deployment with Docker Compose

Deploy the complete stack (FastAPI, Streamlit, Prometheus, Grafana) with a single command:

```bash
cd docker
docker compose up --build
```

Access services:
- **Streamlit UI**: `http://localhost:8501`
- **FastAPI API**: `http://localhost:8000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` *(Default login: admin / admin)*

---

## Automated Testing & Code Quality

The test suite enforces rigorous validation gates across data integrity, model performance regression, and inference contracts:

```bash
# Run full test suite
python -m pytest tests/ -v

# Run code linter and formatting checks
python -m ruff check src/ tests/ dags/
```

Continuous integration runs automatically on every pull request and push to `main` via GitHub Actions.
