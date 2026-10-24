import pandas as pd
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR

# Load the combined district dataset
df = pd.read_csv(
    DATA_DIR / "eduequity_district_dataset_2019_20.csv"
)

print("Original dataset shape:", df.shape)

# Keep only verified enrolment and infrastructure indicators.
# Exclude social-category enrolment columns until their definitions
# and comparability have been verified.

metadata_columns = [
    "State_Code",
    "District_Code",
    "State_Name",
    "District_Name",
]

enrolment_columns = [
    "total_boys",
    "total_girls",
    "total_enrolment",
    "girls_to_boys_ratio",
    "girls_enrolment_share_age_class",
]

# Select infrastructure rates only.
infrastructure_columns = [
    column for column in df.columns
    if column.endswith("_rate")
]

selected_columns = (
    metadata_columns
    + enrolment_columns
    + infrastructure_columns
)

model_df = df[selected_columns].copy()

# Replace infinite values and handle missing values.
model_df = model_df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

# Keep districts with complete modelling features.
feature_columns = (
    enrolment_columns + infrastructure_columns
)

model_df = model_df.dropna(
    subset=feature_columns
).copy()

# Save the clean dataset.
output_file = OUTPUT_DIR / "eduequity_model_dataset_2019_20.csv"

model_df.to_csv(output_file, index=False)

print("\nClean modelling dataset created.")
print("Output file:", output_file)
print("Dataset shape:", model_df.shape)
print("Number of districts:", len(model_df))
print("Number of infrastructure indicators:", len(infrastructure_columns))
print("Missing values in selected features:")
print(model_df[feature_columns].isna().sum().sum())

print("\nFirst five rows:")
print(model_df.head().to_string(index=False))