import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


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
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

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
            transaction_count=("TransactionId", "count")
        ).reset_index()

        # Fill NaN std values (single transaction customers)
        agg["std_amount"] = agg["std_amount"].fillna(0)

        # Merge back customer-level features
        df = df.merge(agg, on="CustomerId", how="left")

        df = add_target_variable(df)

        return df

def create_rfm_target(df):

    data = df.copy()

    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"]
    )

    snapshot_date = (
        data["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = data.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (snapshot_date - x.max()).days
        ),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = (
        rfm.groupby("cluster")
           [["Recency", "Frequency", "Monetary"]]
           .mean()
    )

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
    )

    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0
    )

    return rfm[
        ["CustomerId", "is_high_risk"]
    ]


def add_target_variable(df):

    target_df = create_rfm_target(df)

    df = df.merge(
        target_df,
        on="CustomerId",
        how="left"
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
        "transaction_count"
    ]

    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return pipeline
    return pipeline