import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

folders = [
    DATA_DIR / "enrollment_by_location_social_category" / "EduEquity_Enrolment_CSVs",
    DATA_DIR / "Schools by Availability of Infrastructure and Facilities" / "EduEquity_Infrastructure_CSVs"
]

for folder in folders:
    print("\n" + "=" * 70)
    print("FOLDER:", folder.name)

    files = sorted(folder.glob("*.csv"))
    print("CSV files found:", len(files))

    if not files:
        print("No CSV files found. Check the folder path.")
        continue

    file = files[0]

    try:
        df = pd.read_csv(file, encoding="utf-8-sig", nrows=5)

        print("\nSample file:", file.name)
        print("Columns:")
        print(df.columns.tolist())

        print("\nSample records:")
        print(df.head(3).to_string(index=False))

    except Exception as error:
        print("Could not read file:", error)