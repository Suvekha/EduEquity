
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "EduEquity_Teacher_CSVs"
METADATA_FILE = BASE_DIR / "eduequity_teacher_metadata.csv"

# Load resource metadata
metadata = pd.read_csv(METADATA_FILE, dtype=str)

print("METADATA SUMMARY")
print("-" * 50)
print("Metadata records:", len(metadata))
print("Available columns:")
print(metadata.columns.tolist())

# Inspect the first downloaded CSV
csv_files = sorted(DATA_DIR.glob("*.csv"))

print("\nDOWNLOADED FILE SUMMARY")
print("-" * 50)
print("CSV files found:", len(csv_files))

if not csv_files:
    raise FileNotFoundError("No CSV files found in the download folder.")

sample_file = csv_files[0]
print("\nSample file:", sample_file.name)

sample = pd.read_csv(sample_file, encoding="utf-8-sig")

print("\nSample shape:", sample.shape)
print("\nColumn names:")
print(sample.columns.tolist())

print("\nFirst 5 rows:")
print(sample.head().to_string(index=False))

# Check column consistency across all downloaded files
column_counts = {}
read_errors = []

for index, file in enumerate(csv_files, start=1):
    try:
        df = pd.read_csv(file, encoding="utf-8-sig", nrows=0)
        signature = tuple(df.columns)
        column_counts[signature] = column_counts.get(signature, 0) + 1
    except Exception as error:
        read_errors.append({
            "file": file.name,
            "error": str(error)
        })

print("\nCOLUMN CONSISTENCY")
print("-" * 50)
print("Distinct column structures:", len(column_counts))

for number, (columns, count) in enumerate(
    column_counts.items(), start=1
):
    print(f"\nStructure {number}: {count} files")
    print(list(columns))

print("\nFiles with reading errors:", len(read_errors))

if read_errors:
    pd.DataFrame(read_errors).to_csv(
        BASE_DIR / "teacher_csv_read_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("Error details saved to teacher_csv_read_errors.csv")
