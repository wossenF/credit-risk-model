import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
            lambda x: (snapshot_date - x.max()).days,
        ),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum"),
    ).reset_index()

    if rfm.empty:
        return rfm[["CustomerId"]].assign(
            is_high_risk=pd.Series(dtype="int64")
        )

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    n_clusters = min(3, len(rfm))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("cluster")[
        ["Frequency", "Monetary"]
    ].mean()
    high_risk_cluster = cluster_summary["Frequency"].idxmin()

    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0,
    )

    return rfm[["CustomerId", "is_high_risk"]]

# End of module.

