"""
05_download_environmental_data.py

Fetches / instructs how to obtain the raster environmental layers used in
this project. Nothing here fabricates data - if a layer can't be
auto-downloaded from this machine, the script prints the exact official
URL, expected filename, and target folder, and simply does not proceed
until the file is present.

Layers used:

1. WorldClim v2.1 bioclimatic variables (2.5 arc-minute resolution)
   - BIO1  = Annual Mean Temperature
   - BIO12 = Annual Precipitation
   Official bulk download (no API, direct zip):
     https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip
   Expected files after unzip:
     data/raw/worldclim/wc2.1_2.5m_bio_1.tif   (annual mean temp)
     data/raw/worldclim/wc2.1_2.5m_bio_12.tif  (annual precipitation)

2. Elevation - SRTM-derived global DEM, also distributed by WorldClim at
   matching resolution (simplest reproducible option - avoids the
   USGS EarthExplorer login/tile-mosaic workflow):
     https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_elev.zip
   Expected file:
     data/raw/worldclim/wc2.1_2.5m_elev.tif
   (If you specifically need raw SRTM tiles instead, register at
   https://earthexplorer.usgs.gov/, search "SRTM 1 Arc-Second Global",
   download tiles covering India (lat 6-37N, lon 68-97E), and mosaic them
   with rasterio.merge - a helper for that is in
   scripts/05b_merge_srtm_tiles.py, described in the README.)

3. NDVI (vegetation index) - Sentinel-2 based, NOT auto-downloaded.
   Copernicus Data Space Ecosystem requires an authenticated account and
   an area/date-specific query, which this pipeline cannot do unattended.
   See scripts/06b_add_ndvi.py for the documented, separate workflow using
   MODIS MOD13Q1 (250m, 16-day NDVI composite) via NASA Earthdata as a
   more reproducible alternative to per-scene Sentinel-2 processing.
   The base pipeline in 06_extract_environmental_variables.py runs WITHOUT
   NDVI; NDVI is added as an optional second pass once you have the file.

This script only downloads WorldClim (public, no-login, zip-file endpoints)
automatically. Everything else is manual by design, per your requirement
to never fabricate data we can't actually access.
"""

import io
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WC_DIR = ROOT / "data" / "raw" / "worldclim"
WC_DIR.mkdir(parents=True, exist_ok=True)

WORLDCLIM_FILES = {
    "wc2.1_2.5m_bio.zip": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip",
    "wc2.1_2.5m_elev.zip": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_elev.zip",
}


def download_and_extract(name: str, url: str):
    target_marker = WC_DIR / name.replace(".zip", "_DONE")
    if target_marker.exists():
        print(f"{name}: already extracted, skipping.")
        return

    print(f"Downloading {name} from {url} ...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        content = resp.content
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(WC_DIR)
        target_marker.touch()
        print(f"  Extracted into {WC_DIR}")
    except requests.RequestException as e:
        print(
            f"\n  Could not auto-download {name} ({e}).\n"
            "  MANUAL FALLBACK:\n"
            f"    1. Download: {url}\n"
            f"    2. Unzip into: {WC_DIR}\n"
            "    3. Re-run this script (it will detect the extracted files).\n"
        )


def main():
    print("=== Step 1: WorldClim bioclimatic + elevation rasters ===")
    for name, url in WORLDCLIM_FILES.items():
        download_and_extract(name, url)

    bio1 = list(WC_DIR.glob("*bio_1.tif"))
    bio12 = list(WC_DIR.glob("*bio_12.tif"))
    elev = list(WC_DIR.glob("*elev.tif"))

    print("\n=== Status check ===")
    print(f"Annual mean temperature (bio_1):  {'FOUND: ' + str(bio1[0]) if bio1 else 'MISSING'}")
    print(f"Annual precipitation (bio_12):    {'FOUND: ' + str(bio12[0]) if bio12 else 'MISSING'}")
    print(f"Elevation:                        {'FOUND: ' + str(elev[0]) if elev else 'MISSING'}")

    print(
        "\n=== NDVI ===\n"
        "NDVI is NOT downloaded by this script (requires an authenticated,\n"
        "area/date-specific Sentinel-2 or MODIS query). See the module\n"
        "docstring above and the README section 'Adding NDVI' for the\n"
        "separate, documented workflow. The base pipeline proceeds with\n"
        "temperature, precipitation, and elevation only until you add it.\n"
    )

    if not (bio1 and bio12 and elev):
        print(
            "\nOne or more required rasters are still missing. Follow the "
            "manual fallback instructions above, then re-run this script "
            "before proceeding to 06_extract_environmental_variables.py."
        )


if __name__ == "__main__":
    main()
