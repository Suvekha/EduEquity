
import os
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

METADATA_FILE = "eduequity_enrolment_metadata.csv"
OUTPUT_FOLDER = Path("EduEquity_Enrolment_CSVs")
FAILURE_FILE = "failed_downloads.csv"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(METADATA_FILE)

# Retry temporary network errors and server errors.
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({
    "User-Agent": "EduEquity-Research/1.0"
})

failures = []
downloaded = 0
skipped = 0

total = len(metadata)

for index, row in metadata.iterrows():
    resource_id = row.get("nid")
    title = str(row.get("title", ""))
    url = row.get("datafile_url")

    if pd.isna(url) or not str(url).startswith(
        ("https://", "http://")
    ):
        failures.append({
            "nid": resource_id,
            "title": title,
            "url": url,
            "error": "Missing or invalid URL"
        })
        continue

    # Use the original filename from the published URL.
    filename = unquote(
        os.path.basename(urlparse(str(url)).path)
    )

    if not filename.lower().endswith(".csv"):
        failures.append({
            "nid": resource_id,
            "title": title,
            "url": url,
            "error": "URL does not end in .csv"
        })
        continue

    output_path = OUTPUT_FOLDER / filename

    # Skip previously downloaded non-empty files.
    if output_path.exists() and output_path.stat().st_size > 0:
        skipped += 1
        continue

    try:
        with session.get(
            str(url),
            stream=True,
            timeout=(15, 90)
        ) as response:
            response.raise_for_status()

            temp_path = output_path.with_suffix(".csv.part")

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 64
                ):
                    if chunk:
                        file.write(chunk)

            # Reject empty or obvious HTML responses.
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                temp_path.unlink(missing_ok=True)
                raise ValueError("Downloaded file is empty")

            with open(temp_path, "rb") as file:
                first_bytes = file.read(512).lstrip().lower()

            if (
                first_bytes.startswith(b"<!doctype html")
                or first_bytes.startswith(b"<html")
            ):
                temp_path.unlink(missing_ok=True)
                raise ValueError(
                    "Received HTML instead of a CSV; check the URL"
                )

            temp_path.replace(output_path)

        downloaded += 1

    except Exception as error:
        failures.append({
            "nid": resource_id,
            "title": title,
            "url": url,
            "error": str(error)
        })

    processed = index + 1

    if processed % 25 == 0 or processed == total:
        print(
            f"Processed {processed}/{total} | "
            f"Downloaded: {downloaded} | "
            f"Skipped: {skipped} | "
            f"Failures: {len(failures)}"
        )

    # Keep requests sequential and gentle on the portal.
    time.sleep(0.2)

if failures:
    pd.DataFrame(failures).to_csv(
        FAILURE_FILE,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\nFailure details saved to {FAILURE_FILE}")
else:
    print("\nNo download failures recorded.")

print("\nDownload process finished.")
print(f"New files downloaded: {downloaded}")
print(f"Existing files skipped: {skipped}")
print(f"Failed resources: {len(failures)}")
print(f"Output folder: {OUTPUT_FOLDER.resolve()}")

session.close()
