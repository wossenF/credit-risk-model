# Credit Risk Probability Model for Alternative Data

This repository contains an exploratory analysis notebook and the supporting data for a credit risk modeling workflow. The current EDA notebook reads from `data/raw/data.csv`, which includes the transaction-level fields used for numeric and categorical analysis.

## EDA Notebook

The notebook at `notebooks/eda.ipynb` covers:

- Basic dataset inspection and summary statistics
- Missing-value analysis
- Numeric and categorical distribution plots
- Transaction time feature extraction
- Correlation analysis and fraud-target exploration

The notebook now defines categorical columns before plotting them, so it can run without the earlier `NameError`/`KeyError` issues tied to missing analysis variables and the wrong input file.

## Project Structure

credit-risk-model/

├── .github/workflows/ci.yml

├── data/

│ ├── raw/

│ └── processed/

├── notebooks/

│ └── eda.ipynb

├── src/

│ ├── __init__.py

│ ├── data_processing.py

│ ├── train.py

│ ├── predict.py

│ └── api/

│ ├── main.py

│ └── pydantic_models.py

├── tests/

│ └── test_data_processing.py

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── .gitignore

└── README.md