
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, unquote
import re
from collections import Counter

metadata = pd.read_csv("eduequity_enrolment_metadata.csv")
data_folder = Path("EduEquity_Enrolment_CSVs")

# Count downloaded CSV files.
downloaded_files = list(data_folder.glob("*.csv"))
downloaded_names = {f.name for f in downloaded_files}

# Match each metadata record to its expected local filename.
metadata["filename"] = metadata["datafile_url"].apply(
    lambda url: (
        unquote(Path(urlparse(str(url)).path).name)
        if pd.notna(url) else None
    )
)

metadata["downloaded"] = metadata["filename"].isin(
    downloaded_names
)

# Extract academic years from titles, such as 2019-20 or 2012-13.
def extract_year(title):
    match = re.search(r"\bduring\s+(\d{4}-\d{2})\b", str(title))
    return match.group(1) if match else "Unknown"

metadata["academic_year"] = metadata["title"].apply(extract_year)

print("========== DOWNLOAD SUMMARY ==========")
print("Resources in metadata:", len(metadata))
print("CSV files in folder:", len(downloaded_files))
print("Resources matched to downloaded files:",
      metadata["downloaded"].sum())
print("Resources not matched:",
      (~metadata["downloaded"]).sum())

print("\n========== RESOURCES BY ACADEMIC YEAR ==========")
print(metadata["academic_year"].value_counts().sort_index().to_string())

print("\n========== DOWNLOAD STATUS BY YEAR ==========")
print(
    metadata.groupby("academic_year")["downloaded"]
    .agg(["count", "sum"])
    .rename(columns={
        "count": "resources",
        "sum": "downloaded"
    })
    .to_string()
)

# Save a report for further inspection.
metadata.to_csv(
    "eduequity_data_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved report: eduequity_data_coverage.csv")
