import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "eduequity_model_dataset_2019_20.csv"
)

df = pd.read_csv(DATA_FILE)

print("=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

print("Shape:", df.shape)
print("\nColumn names:")
for column in df.columns:
    print(column)

# Check for duplicate column names.
duplicates = df.columns[df.columns.duplicated()].tolist()
print("\nDuplicate column names:", duplicates)

# Check numerical feature ranges.
feature_columns = [
    column for column in df.columns
    if column.endswith("_rate")
]

print("\n" + "=" * 65)
print("INFRASTRUCTURE RATE CHECK")
print("=" * 65)

print(df[feature_columns].describe().round(3).to_string())

# Rates should normally be between 0 and 1.
invalid_rates = {}

for column in feature_columns:
    invalid_count = (
        (df[column] < 0) | (df[column] > 1)
    ).sum()

    if invalid_count > 0:
        invalid_rates[column] = int(invalid_count)

print("\nInfrastructure rates outside [0, 1]:")
print(invalid_rates if invalid_rates else "None")

print("\nEnrolment indicators:")
print(
    df[
        [
            "total_boys",
            "total_girls",
            "total_enrolment",
            "girls_to_boys_ratio",
            "girls_enrolment_share_age_class",
        ]
    ].describe().round(3).to_string()
)