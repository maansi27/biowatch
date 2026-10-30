"""
01_get_species_key.py

Resolve the GBIF backbone taxon key for a species name using the official
GBIF species-matching API. Never hard-code an unverified key: this script
always queries the API and prints exactly what GBIF returns.

GBIF API used:
    GET https://api.gbif.org/v1/species/match?name=<name>

Output:
    Prints the matched scientific name, accepted name, rank, status and key.
    Writes the resolved key to data/raw/species_key.json for downstream
    scripts to reuse.

Note: This has already been verified manually for Lantana camara during
development of this pipeline: GBIF backbone taxon key = 2925303
(accepted species, kingdom Plantae), confirmed at
https://www.gbif.org/species/2925303
This script re-derives it independently so the pipeline stays reproducible
even if you point it at a different species.
"""

import json
import sys
from pathlib import Path

import requests

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"

# Change this if you want to build the dataset for a different invasive
# species from the GRIIS India checklist (see README for the fallback list).
SPECIES_NAME = "Lantana camara"

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "species_key.json"


def resolve_species_key(name: str) -> dict:
    params = {"name": name, "kingdom": "Plantae", "strict": "false"}
    resp = requests.get(GBIF_MATCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("matchType") == "NONE" or "usageKey" not in data:
        print(f"ERROR: GBIF could not match the name '{name}'.")
        print(json.dumps(data, indent=2))
        sys.exit(1)

    return data


def main():
    print(f"Querying GBIF species match API for: '{SPECIES_NAME}'")
    match = resolve_species_key(SPECIES_NAME)

    print("\n--- GBIF match result ---")
    print(f"Query name:          {SPECIES_NAME}")
    print(f"Matched name:        {match.get('scientificName')}")
    print(f"Canonical name:      {match.get('canonicalName')}")
    print(f"Accepted name:       {match.get('species', match.get('scientificName'))}")
    print(f"Rank:                {match.get('rank')}")
    print(f"Status:               {match.get('status')}")
    print(f"Match type:          {match.get('matchType')}")
    print(f"Confidence:          {match.get('confidence')}")
    print(f"GBIF usageKey:       {match.get('usageKey')}")
    print(f"GBIF speciesKey:     {match.get('speciesKey')}")

    result = {
        "query_name": SPECIES_NAME,
        "scientificName": match.get("scientificName"),
        "canonicalName": match.get("canonicalName"),
        "status": match.get("status"),
        "rank": match.get("rank"),
        "usageKey": match.get("usageKey"),
        "speciesKey": match.get("speciesKey"),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved resolved key to {OUT_FILE}")
    print(f"\nUse taxonKey = {match.get('speciesKey') or match.get('usageKey')} in script 02.")


if __name__ == "__main__":
    main()
