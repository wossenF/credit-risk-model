import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.rfm_target import create_rfm_target


# ===============================
# 1. FEATURE ENGINEERING
# ===============================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert datetime
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        # Time features
        df["hour"] = df["TransactionStartTime"].dt.hour
        df["day"] = df["TransactionStartTime"].dt.day
        df["month"] = df["TransactionStartTime"].dt.month
        df["year"] = df["TransactionStartTime"].dt.year

        # Aggregate features per customer
        agg = df.groupby("CustomerId").agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            transaction_count=("TransactionId", "count"),
        ).reset_index()

        # Fill NaN std values (single transaction customers)
        agg["std_amount"] = agg["std_amount"].fillna(0)

        # Merge back customer-level features
        df = df.merge(agg, on="CustomerId", how="left")
        return df


def add_target_variable(df):
    target_df = create_rfm_target(df)

    df = df.merge(
        target_df,
        on="CustomerId",
        how="left",
    )
    return df


# ===============================
# 2. PIPELINE BUILDER
# ===============================
def build_pipeline():
    numeric_features = [
        "Amount",
        "Value",
        "hour",
        "day",
        "month",
        "year",
        "total_amount",
        "avg_amount",
        "std_amount",
        "transaction_count",
    ]

    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
    ])

    return pipeline

# End of module.

