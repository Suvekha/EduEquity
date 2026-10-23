
from pathlib import Path
from collections import Counter
import pandas as pd

DATA_FOLDER = Path("EduEquity_Enrolment_CSVs")

files = list(DATA_FOLDER.glob("*.csv"))

schema_counts = Counter()
schema_examples = {}
file_errors = []

for index, file_path in enumerate(files, start=1):
    try:
        # Read only the header to inspect the column structure.
        columns = tuple(
            pd.read_csv(file_path, nrows=0).columns.astype(str)
        )

        schema_counts[columns] += 1
        schema_examples.setdefault(columns, file_path.name)

    except Exception as error:
        file_errors.append({
            "file": file_path.name,
            "error": str(error)
        })

    if index % 500 == 0 or index == len(files):
        print(f"Inspected {index}/{len(files)} files")

print("\n========== SCHEMA SUMMARY ==========")
print("Total CSV files:", len(files))
print("Distinct column layouts:", len(schema_counts))

for number, (columns, count) in enumerate(
    schema_counts.most_common(), start=1
):
    print(f"\nSchema {number}: {count} files")
    print("Example file:", schema_examples[columns])
    print("Columns:")
    for column in columns:
        print(" -", column)

print("\n========== FILE READ ERRORS ==========")
print("Files with header errors:", len(file_errors))

if file_errors:
    pd.DataFrame(file_errors).to_csv(
        "eduequity_schema_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("Saved: eduequity_schema_errors.csv")
