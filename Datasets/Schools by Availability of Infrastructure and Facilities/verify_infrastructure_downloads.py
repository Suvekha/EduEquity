
from pathlib import Path
import pandas as pd

folder = Path("EduEquity_Infrastructure_CSVs")
metadata_file = Path("eduequity_infrastructure_metadata.csv")
retry_file = Path("infrastructure_retry_failures.csv")

metadata = pd.read_csv(metadata_file)
files = list(folder.glob("*.csv"))

print("Expected resources:", len(metadata))
print("CSV files currently present:", len(files))
print("Resources not downloaded:", len(metadata) - len(files))

if retry_file.exists():
    remaining = pd.read_csv(retry_file)
    print("Failures remaining after retry:", len(remaining))
    if not remaining.empty:
        print("\nRemaining failures:")
        print(remaining[["title", "error"]].to_string(index=False))
