import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/processed/processed_data.csv")

X = df.drop(columns=["is_high_risk"])
y = df["is_high_risk"]

# remove any leftover IDs if exist
X = X.drop(columns=["CustomerId", "TransactionId"], errors="ignore")

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# METRICS FUNCTION
# =========================
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds)
    }

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("credit-risk-model")

# =========================
# MODEL 1 - LOGISTIC REGRESSION
# =========================
with mlflow.start_run(run_name="LogisticRegression"):

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)

    metrics = evaluate(log_model, X_test, y_test)

    mlflow.log_params({"model": "LogisticRegression"})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(log_model, "model")

# =========================
# MODEL 2 - RANDOM FOREST
# =========================
with mlflow.start_run(run_name="RandomForest"):

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1")
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    metrics = evaluate(best_rf, X_test, y_test)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(best_rf, "model", registered_model_name="CreditRiskModel")

print("Training complete!")