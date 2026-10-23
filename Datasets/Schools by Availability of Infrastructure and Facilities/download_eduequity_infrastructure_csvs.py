
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

METADATA_FILE = Path("eduequity_infrastructure_metadata.csv")
OUTPUT_FOLDER = Path("EduEquity_Infrastructure_CSVs")
FAILURE_FILE = Path("infrastructure_failed_downloads.csv")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(METADATA_FILE)

required_columns = {"nid", "title", "datafile_url"}
if not required_columns.issubset(metadata.columns):
    raise ValueError(
        f"Metadata must contain these columns: {required_columns}"
    )

session = requests.Session()
session.headers.update({
    "User-Agent": "EduEquity-Research/1.0"
})

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

failures = []
success_count = 0
skipped_count = 0

for index, row in metadata.iterrows():
    url = row["datafile_url"]

    if pd.isna(url) or not str(url).startswith(("http://", "https://")):
        failures.append({
            "nid": row["nid"],
            "title": row["title"],
            "url": url,
            "error": "Missing or invalid URL"
        })
        continue

    # Use the filename from the source URL.
    filename = unquote(Path(urlparse(str(url)).path).name)

    if not filename:
        filename = f"resource_{row['nid']}.csv"

    # Prevent unsafe path characters from being used as filenames.
    filename = Path(filename).name
    output_path = OUTPUT_FOLDER / filename
    temp_path = OUTPUT_FOLDER / (filename + ".part")

    if output_path.exists() and output_path.stat().st_size > 0:
        skipped_count += 1
        continue

    try:
        with session.get(
            str(url),
            stream=True,
            timeout=(20, 120)
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            first_chunk = True
            bytes_written = 0

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=64 * 1024
                ):
                    if not chunk:
                        continue

                    if first_chunk:
                        first_chunk = False
                        # Catch common HTML error/login pages.
                        if (
                            b"<html" in chunk[:1000].lower()
                            or b"<!doctype html" in chunk[:1000].lower()
                        ):
                            raise ValueError(
                                "Server returned HTML instead of a CSV"
                            )

                    file.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written == 0:
                raise ValueError("Downloaded file is empty")

            temp_path.replace(output_path)

        success_count += 1

    except Exception as error:
        if temp_path.exists():
            temp_path.unlink()

        failures.append({
            "nid": row["nid"],
            "title": row["title"],
            "url": url,
            "error": str(error)
        })

    processed = index + 1
    if processed % 25 == 0 or processed == len(metadata):
        print(
            f"Processed {processed}/{len(metadata)} | "
            f"Downloaded this run: {success_count} | "
            f"Skipped existing: {skipped_count} | "
            f"Failures: {len(failures)}"
        )

    time.sleep(0.05)

pd.DataFrame(failures).to_csv(
    FAILURE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDownload run finished.")
print(f"New downloads: {success_count}")
print(f"Already-existing files skipped: {skipped_count}")
print(f"Failures this run: {len(failures)}")
print(f"Dataset folder: {OUTPUT_FOLDER.resolve()}")
print(f"Failure log: {FAILURE_FILE.resolve()}")
