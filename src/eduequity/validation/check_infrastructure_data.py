import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

FOLDER = (
    ROOT_DIR
    / "data"
    / "raw"
    / "infrastructure"
    / "Schools by Availability of Infrastructure and Facilities"
    / "EduEquity_Infrastructure_CSVs"
)

files = sorted(
    file for file in FOLDER.glob("*.csv")
    if "19-20" in file.name or "2019-20" in file.name
)

print("2019–20 files:", len(files))

if not files:
    raise FileNotFoundError("No 2019–20 infrastructure CSV files found.")

file = files[0]
df = pd.read_csv(file, encoding="utf-8-sig")

print("\nFile:", file.name)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nRecords where facility values exceed total schools:")

facility_columns = [
    "Building",
    "Functional_Electricity",
    "Functional_Boy_Toilet",
    "Functional_Girl_Toilet",
    "Functional_Drinking_Water",
    "Internet",
    "Computer_Available",
]

for column in facility_columns:
    if column not in df.columns:
        continue

    values = pd.to_numeric(df[column], errors="coerce")
    total = pd.to_numeric(
        df["Total_Number_of_Schools"], errors="coerce"
    )

    invalid = df[values > total]

    print(f"\n{column}: {len(invalid)} suspicious rows")

    if not invalid.empty:
        print(
            invalid[
                [
                    "District_Code",
                    "Block_Code",
                    "Total_Number_of_Schools",
                    column,
                ]
            ].head(5).to_string(index=False)
        )

print("\nMissing values per column:")
print(df.isna().sum().to_string())