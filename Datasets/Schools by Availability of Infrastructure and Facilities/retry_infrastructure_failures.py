
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

BASE_FOLDER = Path.cwd()
METADATA_FILE = BASE_FOLDER / "eduequity_infrastructure_metadata.csv"
FAILURE_FILE = BASE_FOLDER / "infrastructure_failed_downloads.csv"
OUTPUT_FOLDER = BASE_FOLDER / "EduEquity_Infrastructure_CSVs"
NEW_FAILURE_FILE = BASE_FOLDER / "infrastructure_retry_failures.csv"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

if not FAILURE_FILE.exists():
    raise FileNotFoundError(f"Missing failure log: {FAILURE_FILE}")

failed = pd.read_csv(FAILURE_FILE)
metadata = pd.read_csv(METADATA_FILE)

# Get the original resource details for failed IDs.
failed_ids = set(failed["nid"].astype(str))
metadata["nid"] = metadata["nid"].astype(str)
retry_rows = metadata[metadata["nid"].isin(failed_ids)]

session = requests.Session()
session.headers.update({"User-Agent": "EduEquity-Research/1.0"})

retry = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

remaining_failures = []
success = 0

for _, row in retry_rows.iterrows():
    url = row["datafile_url"]
    filename = unquote(Path(urlparse(str(url)).path).name)
    filename = Path(filename).name or f"resource_{row['nid']}.csv"
    output_path = OUTPUT_FOLDER / filename
    temp_path = OUTPUT_FOLDER / (filename + ".part")

    try:
        with session.get(
            url, stream=True, timeout=(20, 120)
        ) as response:
            response.raise_for_status()

            written = 0
            first_chunk = True

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue

                    if first_chunk:
                        first_chunk = False
                        beginning = chunk[:1000].lower()
                        if b"<html" in beginning or b"<!doctype html" in beginning:
                            raise ValueError("Received HTML instead of CSV")

                    f.write(chunk)
                    written += len(chunk)

            if written == 0:
                raise ValueError("Downloaded file is empty")

            temp_path.replace(output_path)

        success += 1
        print(f"Downloaded: {filename}")

    except Exception as error:
        if temp_path.exists():
            temp_path.unlink()

        remaining_failures.append({
            "nid": row["nid"],
            "title": row["title"],
            "datafile_url": url,
            "error": str(error)
        })
        print(f"Failed again: {row['title']} — {error}")

    time.sleep(0.3)

pd.DataFrame(remaining_failures).to_csv(
    NEW_FAILURE_FILE, index=False, encoding="utf-8-sig"
)

print("\nRetry finished.")
print(f"Successful retries: {success}")
print(f"Still failing: {len(remaining_failures)}")
print(f"Retry failure log: {NEW_FAILURE_FILE}")
