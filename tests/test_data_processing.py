import pandas as pd
from src.data_processing import FeatureEngineer, create_rfm_target


def test_feature_engineering():

    df = pd.DataFrame({
        "CustomerId": ["C1"],
        "TransactionId": ["T1"],
        "Amount": [100],
        "Value": [100],
        "TransactionStartTime": ["2025-01-01"]
    })

    transformed = FeatureEngineer().fit_transform(df)

    assert "hour" in transformed.columns


def test_rfm_target():

    df = pd.DataFrame({
        "CustomerId": ["C1", "C1"],
        "TransactionId": ["T1", "T2"],
        "Amount": [100, 200],
        "TransactionStartTime": ["2025-01-01", "2025-01-02"]
    })

    result = create_rfm_target(df)

    assert "CustomerId" in result.columns
