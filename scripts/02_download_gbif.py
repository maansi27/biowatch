"""
02_download_gbif.py

Download ALL GBIF occurrence records for the target species within India,
using the official GBIF occurrence search API with correct pagination.

GBIF API used:
    GET https://api.gbif.org/v1/occurrence/search
        ?taxonKey=<key>
        &country=IN
        &hasCoordinate=true
        &limit=300 (max page size)
        &offset=<n>  (paged, GBIF search endpoint caps at offset+limit <= 100000
                      records; for larger datasets use the async /occurrence/download
                      endpoint instead - see NOTE at bottom of this file)

This script does NOT stop at an arbitrary record count. It keeps paging
until GBIF reports endOfRecords = true (or the 100,000-record search cap
is hit, in which case it prints a warning to use occ_download instead).

Output:
    data/raw/gbif_lantana_india.csv  -- every field GBIF returned, one row
                                         per occurrence record (raw, uncleaned)
"""

import csv
import json
import time
from pathlib import Path

import requests

OCC_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
PAGE_SIZE = 300  # GBIF max per page for the search endpoint

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KEY_FILE = RAW_DIR / "species_key.json"
OUT_CSV = RAW_DIR / "gbif_lantana_india.csv"

FIELDS = [
    "key", "occurrenceID", "gbifID", "scientificName", "species",
    "taxonKey", "speciesKey", "kingdom", "family", "genus",
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
    "year", "month", "day", "eventDate",
    "basisOfRecord", "datasetKey", "datasetName", "institutionCode",
    "recordedBy", "country", "countryCode", "stateProvince",
    "locality", "issues", "license", "publishingCountry",
]


def load_taxon_key() -> int:
    if not KEY_FILE.exists():
        raise SystemExit(
            "species_key.json not found. Run 01_get_species_key.py first."
        )
    with open(KEY_FILE) as f:
        data = json.load(f)
    key = data.get("speciesKey") or data.get("usageKey")
    if not key:
        raise SystemExit("No usable taxon key found in species_key.json")
    return int(key)


def fetch_page(taxon_key: int, offset: int) -> dict:
    params = {
        "taxonKey": 2925303,
        "country": "IN",
        "hasCoordinate": "true",
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    for attempt in range(5):
        try:
            resp = requests.get(OCC_SEARCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  Request failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise SystemExit("Failed to fetch a page from GBIF after 5 retries.")


def flatten_record(rec: dict) -> dict:
    row = {}
    for field in FIELDS:
        val = rec.get(field, "")
        if isinstance(val, list):
            val = "; ".join(str(v) for v in val)
        row[field] = val
    return row


def main():
    taxon_key = load_taxon_key()
    print(f"Downloading GBIF occurrences for taxonKey={taxon_key}, country=IN, hasCoordinate=true")

    offset = 0
    total_written = 0
    end_of_records = False
    reported_count = None

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        while not end_of_records:
            page = fetch_page(taxon_key, offset)

            if reported_count is None:
                reported_count = page.get("count", 0)
                print(f"GBIF reports {reported_count} total matching records.")

            results = page.get("results", [])
            for rec in results:
                writer.writerow(flatten_record(rec))
                total_written += 1

            end_of_records = page.get("endOfRecords", True)
            offset += PAGE_SIZE

            print(f"  Fetched {total_written}/{reported_count} records "
                  f"(offset={offset})...")

            if offset >= 100000 and not end_of_records:
                print(
                    "\nWARNING: reached the GBIF search-endpoint cap "
                    "(offset+limit must stay <= 100000). If reported_count "
                    "exceeds 100,000, use GBIF's asynchronous "
                    "/occurrence/download endpoint (occ_download in rgbif / "
                    "pygbif) with a GBIF account instead of /occurrence/search."
                )
                break

            time.sleep(0.3)  # be polite to the API

    print(f"\nDone. Wrote {total_written} raw records to {OUT_CSV}")
    print(f"GBIF-reported total for this query: {reported_count}")
    if total_written != reported_count:
        print(
            "NOTE: written count differs from GBIF's reported count - this "
            "can happen if the search hit the 100k offset cap, or if GBIF's "
            "index changed between the count and the paged fetch. Re-run "
            "if the gap is large."
        )


if __name__ == "__main__":
    main()

# NOTE on scale:
# For most single-species, single-country queries (including Lantana camara
# in India) the /occurrence/search endpoint with pagination is sufficient -
# GBIF typically returns well under 100,000 India records for this species.
# If your species/country combination is larger, switch to occ_download()
# from the pygbif or rgbif packages, which requires a free GBIF account
# (https://www.gbif.org/user/profile) and an email/user/password, and
# produces a Darwin Core Archive you then unzip and load instead.
