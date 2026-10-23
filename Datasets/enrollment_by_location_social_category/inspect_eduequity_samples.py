
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("EduEquity_Enrolment_CSVs")

sample_names = [
    "UDISE_plus-15-16_enrol_adilabad_tel.csv",
    "UDISE_plus-12-13_enrol_agra_up.csv",
    "UDISE_plus-19-20_enrol_adilabad_tel.csv"
]

for filename in sample_names:
    path = DATA_FOLDER / filename

    print("\n" + "=" * 80)
    print("FILE:", filename)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path, low_memory=False)

    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nAcademic years in file:")
    if "ac_year" in df.columns:
        print(df["ac_year"].value_counts(dropna=False).to_string())

    print("\nDistinct district names:")
    if "district_name" in df.columns:
        print(df["district_name"].nunique(dropna=True))
        print(df["district_name"].dropna().unique()[:10])

    print("\nMissing values in key columns:")
    key_columns = [
        "district_name", "block_name", "udise_block_name",
        "caste_name", "sch_mgmt_name", "category_name",
        "school_category", "loc_name"
    ]
    available = [c for c in key_columns if c in df.columns]
    if available:
        print(df[available].isna().sum().to_string())

    print("\nFirst few dimension combinations:")
    dimensions = [
        c for c in [
            "district_name", "block_name", "udise_block_name",
            "sch_category_id", "category_name", "school_category",
            "sch_mgmt_id", "sch_mgmt_name", "loc_name",
            "caste_id", "caste_name"
        ] if c in df.columns
    ]
    if dimensions:
        print(df[dimensions].head(10).to_string(index=False))
