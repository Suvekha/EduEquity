import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

folders = {
    "Social Category": (
        BASE_DIR / "enrollment_by_location_social_category"
        / "EduEquity_Enrolment_CSVs"
    ),
    "Infrastructure": (
        BASE_DIR / "Schools by Availability of Infrastructure and Facilities"
        / "EduEquity_Infrastructure_CSVs"
    )
}

for dataset_name, folder in folders.items():
    print("\n" + "=" * 80)
    print(dataset_name.upper())
    print("=" * 80)

    # Find files for the 2019–20 academic year
    files = sorted(
        file for file in folder.glob("*.csv")
        if "19-20" in file.name or "2019-20" in file.name
    )

    print("2019–20 files found:", len(files))

    if not files:
        print("No matching files. Check the filenames.")
        continue

    file = files[0]
    df = pd.read_csv(file, encoding="utf-8-sig", nrows=100)

    print("\nSample file:", file.name)
    print("\nExact column names:")

    for column in df.columns:
        print(repr(column))

    print("\nData types:")
    print(df.dtypes.to_string())

    if dataset_name == "Social Category":
        if "caste_name" in df.columns:
            print("\nSocial categories:")
            print(df["caste_name"].value_counts(dropna=False))

    else:
        facility_columns = [
            column for column in df.columns
            if column not in [
                "Academic_Year", "State_Code", "State_Name",
                "District_Code", "District_Name", "Block_Code",
                "Udise_Block_Name", "School_Category_Id",
                "School_Category_Name", "School_Management_Id",
                "School_Management_Name", "Location",
                "School_Type_Id", "School_Type",
                "Total_Number_of_Schools"
            ]
        ]

        print("\nInfrastructure values in the sample:")
        for column in facility_columns:
            print(
                f"{column}:",
                df[column].value_counts(dropna=False).head(10).to_dict()
            )

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))