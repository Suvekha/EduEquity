
import pandas as pd
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# 1. Define folder paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "EduEquity_Teacher_CSVs"
ERROR_FILE = BASE_DIR / "teacher_csv_read_errors.csv"

# --------------------------------------------------
# 2. Find all downloaded CSV files
# --------------------------------------------------

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 60)
print("EDUEQUITY - TEACHER DATA STRUCTURE CHECK")
print("=" * 60)

print("Data folder:", DATA_DIR)
print("CSV files found:", len(csv_files))

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found. Check the EduEquity_Teacher_CSVs folder."
    )

# --------------------------------------------------
# 3. Initialize variables
# --------------------------------------------------

column_structures = Counter()
read_errors = []

years = set()
districts = set()
state_district_pairs = set()

total_rows = 0

# --------------------------------------------------
# 4. Inspect every CSV file
# --------------------------------------------------

for index, file in enumerate(csv_files, start=1):

    try:
        # Read only the header to identify column structure
        header = pd.read_csv(
            file,
            encoding="utf-8-sig",
            nrows=0
        )

        columns = tuple(header.columns)
        column_structures[columns] += 1

        # Read the complete file for coverage statistics
        df = pd.read_csv(
            file,
            encoding="utf-8-sig"
        )

        total_rows += len(df)

        # Collect academic years
        if "Academic_Year" in df.columns:
            years.update(
                df["Academic_Year"]
                .dropna()
                .astype(str)
                .unique()
            )

        # Collect district codes
        if "District_Code" in df.columns:
            districts.update(
                df["District_Code"]
                .dropna()
                .astype(str)
                .unique()
            )

        # Collect unique state-district combinations
        required_columns = {"State_Code", "District_Code"}

        if required_columns.issubset(df.columns):
            pairs = df[
                ["State_Code", "District_Code"]
            ].dropna()

            for state_code, district_code in zip(
                pairs["State_Code"],
                pairs["District_Code"]
            ):
                state_district_pairs.add(
                    (str(state_code), str(district_code))
                )

    except Exception as error:
        read_errors.append({
            "file": file.name,
            "error": str(error)
        })

    # Display progress
    if index % 250 == 0 or index == len(csv_files):
        print(f"Checked {index}/{len(csv_files)} files")

# --------------------------------------------------
# 5. Display column structures
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN STRUCTURE ANALYSIS")
print("=" * 60)

print("Distinct column structures:", len(column_structures))

structures = list(column_structures.items())

for number, (columns, count) in enumerate(
    structures,
    start=1
):
    print(f"\nStructure {number}")
    print("-" * 40)

    print("Number of files:", count)
    print("Number of columns:", len(columns))

    print("\nColumn names:")

    for column in columns:
        print(" -", column)

# --------------------------------------------------
# 6. Compare the structures
# --------------------------------------------------

if len(structures) >= 2:

    reference_columns = set(structures[0][0])

    print("\n" + "=" * 60)
    print("COLUMN DIFFERENCES")
    print("=" * 60)

    for number, (columns, count) in enumerate(
        structures[1:],
        start=2
    ):
        current_columns = set(columns)

        print(f"\nStructure 1 compared with Structure {number}")

        only_in_reference = sorted(
            reference_columns - current_columns
        )

        only_in_current = sorted(
            current_columns - reference_columns
        )

        print("\nColumns only in Structure 1:")

        if only_in_reference:
            for column in only_in_reference:
                print(" -", column)
        else:
            print("None")

        print(f"\nColumns only in Structure {number}:")

        if only_in_current:
            for column in only_in_current:
                print(" -", column)
        else:
            print("None")

# --------------------------------------------------
# 7. Display coverage statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA COVERAGE")
print("=" * 60)

print("Total CSV files discovered:", len(csv_files))
print("Total rows read successfully:", total_rows)

print("\nAcademic years found:")

if years:
    for year in sorted(years):
        print(" -", year)
else:
    print("No academic-year values found.")

print("\nUnique district codes:", len(districts))

print(
    "Unique state-district combinations:",
    len(state_district_pairs)
)

# --------------------------------------------------
# 8. Save file-reading errors, if any
# --------------------------------------------------

print("\n" + "=" * 60)
print("READING ERRORS")
print("=" * 60)

print("Files with reading errors:", len(read_errors))

if read_errors:

    pd.DataFrame(read_errors).to_csv(
        ERROR_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("Error report saved to:", ERROR_FILE)

    print("\nFirst few errors:")

    for error in read_errors[:10]:
        print(error["file"], ":", error["error"])

else:
    print("All discovered CSV files were read successfully.")

    if ERROR_FILE.exists():
        ERROR_FILE.unlink()

# --------------------------------------------------
# 9. Finish
# --------------------------------------------------

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)
