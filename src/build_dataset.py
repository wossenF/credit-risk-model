from src.data_processing import build_pipeline
from src.rfm_target import create_rfm_target
from pathlib import Path
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

df = pd.read_csv(RAW_DATA_PATH)

# 1. CREATE TARGET FIRST (RAW DATA)
target_df = create_rfm_target(df)

# Keep the customer-level target aligned with each transaction row.
df_with_target = df.merge(target_df, on="CustomerId", how="left")

# 2. BUILD FEATURES
pipeline = build_pipeline()
X = pipeline.fit_transform(df)

# 3. CONVERT TO DATAFRAME
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

if sparse.issparse(df):
	df = df.toarray()

df = pd.DataFrame(df, columns=feature_names)

# 4. MERGE TARGET
final_df = pd.concat(
	[
		df,
		df_with_target[["is_high_risk"]].reset_index(drop=True)
	],
	axis=1
)

final_df.to_csv(OUTPUT_PATH, index=False)

print("SUCCESS")