# Credit Risk Probability Model for Alternative Data

This repository contains the full workflow for a credit risk modeling project: exploratory analysis, feature engineering, model training, MLflow model registration, and a FastAPI deployment layer.

## What’s Included

- `notebooks/eda.ipynb` for exploratory analysis on the raw transaction data.
- `src/data_processing.py` and `src/rfm_target.py` for feature engineering and proxy target creation.
- `src/train.py` for training and logging candidate models to MLflow.
- `src/api/main.py` and `src/api/pydantic_models.py` for the prediction API.
- `Dockerfile` and `docker-compose.yml` for containerized execution.
- `.github/workflows/ci.yml` for automated linting and tests on pushes to `main`.

## Data

The project uses `data/raw/data.csv` as the transaction dataset. The file `data/raw/Xente_Variable_Definitions.csv` contains the variable definitions for reference.

## EDA Notebook

The notebook at `notebooks/eda.ipynb` covers:

- Basic dataset inspection and summary statistics
- Missing-value analysis
- Numeric and categorical distribution plots
- Transaction time feature extraction
- Correlation analysis and fraud-target exploration

## Training and MLflow

Run the dataset builder first to produce the processed training table:

```powershell
python -m src.build_dataset
```

Then train and log the models:

```powershell
python -m src.train
```

The best model is registered in MLflow as `CreditRiskModel`, and the API loads the latest registered version by default.

## FastAPI Service

Start the API locally with Uvicorn:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Key endpoints:

- `GET /health` returns the model loading status.
- `POST /predict` accepts a JSON payload shaped like `{"features": {...}}` and returns a risk probability.

## Containerized Run

Build and run with Docker Compose:

```powershell
docker compose up --build
```

The container expects an accessible MLflow model registry or local `mlruns` directory mounted into the service.

## Continuous Integration

The workflow in `.github/workflows/ci.yml` runs on pushes to `main` and executes:

- `flake8` for code quality checks
- `pytest` for the unit test suite

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── api/
│   ├── build_dataset.py
│   ├── data_processing.py
│   ├── rfm_target.py
│   └── train.py
├── tests/
│   └── test_data_processing.py
├── requirements.txt
└── README.md
```