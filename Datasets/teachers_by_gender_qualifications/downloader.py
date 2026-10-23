
import pandas as pd
import requests
import time
import re
from pathlib import Path
from urllib.parse import urlparse, unquote

# Folder containing this Python script and metadata
BASE_DIR = Path(__file__).resolve().parent

METADATA_FILE = BASE_DIR / "eduequity_teacher_metadata.csv"
DOWNLOAD_DIR = BASE_DIR / "EduEquity_Teacher_CSVs"
FAILED_FILE = BASE_DIR / "failed_teacher_downloads.csv"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Read the metadata
df = pd.read_csv(METADATA_FILE, dtype=str)

if "datafile_url" not in df.columns:
    raise ValueError("The metadata file does not contain 'datafile_url'.")

if "nid" not in df.columns:
    raise ValueError("The metadata file does not contain 'nid'.")

# Keep records that have a download URL
df = df.dropna(subset=["datafile_url"])
df = df[df["datafile_url"].str.startswith("http", na=False)]

print("Resources with download URLs:", len(df))
print("Download folder:", DOWNLOAD_DIR)

failed_downloads = []
total = len(df)
downloaded = 0
skipped = 0

for index, row in df.iterrows():
    resource_id = str(row["nid"]).strip()
    url = str(row["datafile_url"]).strip()
    title = str(row.get("title", ""))

    # Use the original filename, prefixed by the resource ID.
    original_name = unquote(Path(urlparse(url).path).name)

    if not original_name:
        original_name = f"teacher_resource_{resource_id}.csv"

    # Remove characters that are unsafe in Windows filenames.
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", original_name)
    filename = f"{resource_id}_{safe_name}"
    output_path = DOWNLOAD_DIR / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        skipped += 1
    else:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            content = response.content

            # Avoid saving empty or obvious HTML error pages as CSV files.
            content_start = content.lstrip()[:1000].lower()
            if not content:
                raise ValueError("Downloaded file is empty.")

            if (
                content_start.startswith(b"<!doctype html")
                or content_start.startswith(b"<html")
            ):
                raise ValueError("The URL returned HTML instead of a CSV file.")

            output_path.write_bytes(content)
            downloaded += 1

        except Exception as error:
            failed_downloads.append({
                "nid": resource_id,
                "title": title,
                "url": url,
                "error": str(error)
            })

    completed = downloaded + skipped + len(failed_downloads)

    if completed % 50 == 0 or completed == total:
        print(
            f"Processed {completed}/{total} | "
            f"Downloaded: {downloaded} | "
            f"Skipped: {skipped} | "
            f"Failed: {len(failed_downloads)}"
        )

    time.sleep(0.1)

# Save failed downloads for later retry or documentation.
if failed_downloads:
    pd.DataFrame(failed_downloads).to_csv(
        FAILED_FILE,
        index=False,
        encoding="utf-8-sig"
    )
    print("\nFailed download details saved to:", FAILED_FILE)
else:
    # Remove an old failure report if this run had no failures.
    if FAILED_FILE.exists():
        FAILED_FILE.unlink()
    print("\nAll attempted downloads succeeded.")

print("\nDownload process finished.")
print("Total metadata records with URLs:", total)
print("New files downloaded:", downloaded)
print("Existing files skipped:", skipped)
print("Failed downloads:", len(failed_downloads))
print("CSV files are stored in:", DOWNLOAD_DIR)
