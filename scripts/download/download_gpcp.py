"""Download GPCP V3.2 monthly precipitation (GPCPMON, 0.5 deg, 1983-01 to 2023-12) from NASA GES DISC.
Dataset: GPCP Precipitation Level 3 Monthly 0.5-Degree V3.2, Huffman et al., doi:10.5067/MEASURES/GPCP/DATA304
The file URLs are listed by the NASA CMR granule search (https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html).

Local layout:  OUTPUT_ROOT/GPCPMON_L3_<yyyymm>_V3.2.nc4   (one file per month, ~6.5 MB, 492 files, ~3.2 GB)
OUTPUT_ROOT is SCRATCH_ROOT/gpcp-v<VERSION> (config/paths.yml); move it to OBS_ROOT by hand once complete.
Each file is written to <name>.part and renamed only once it is complete and starts with the HDF5
signature (netCDF-4), so a file that exists is complete: re-running is safe and retries only the
missing files.

Needs an Earthdata Login account (https://urs.earthdata.nasa.gov) with:
  - the "NASA GESDISC DATA ARCHIVE" application approved once in the profile (Applications >
    Authorized Apps), otherwise every file fails with 401 "Pre authorization required";
  - ~/.netrc (chmod 600) with the line
        machine urs.earthdata.nasa.gov login <username> password <password>
    which requests reads when GES DISC redirects the download to Earthdata Login.
Run:  pixi run python scripts/download/download_gpcp.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from etunc.config import SCRATCH_ROOT

SHORT_NAME = "GPCPMON"
VERSION = "3.3"
OUTPUT_ROOT = SCRATCH_ROOT / f"gpcp-v{VERSION}"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
MAX_WORKERS = 3  # parallel downloads, kept small on a shared login node
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"  # first 8 bytes of every netCDF-4 file


def list_download_urls(short_name, version):
    """HTTPS URLs of every granule of a collection, following CMR's search-after pagination."""
    urls = []
    headers = {}
    while True:
        params = {"short_name": short_name, "version": version, "page_size": 2000}
        response = requests.get(CMR_GRANULES_URL, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        for item in response.json()["items"]:
            # "GET DATA" is the HTTPS link; the s3:// "GET DATA VIA DIRECT ACCESS" works only in AWS us-west-2
            urls += [u["URL"] for u in item["umm"].get("RelatedUrls", []) if u.get("Type") == "GET DATA"]
        # CMR returns a search-after token while more pages remain
        search_after = response.headers.get("CMR-Search-After")
        if not search_after:
            return urls
        headers["CMR-Search-After"] = search_after


def download_file(url):
    """Download one file to OUTPUT_ROOT; return "downloaded" or "skipped" (file exists), raise on failure."""
    path = OUTPUT_ROOT / url.rsplit("/", 1)[-1]
    if path.exists():
        return "skipped"
    part = path.with_name(path.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with open(part, "rb") as f:
            if f.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
                raise ValueError("not a netCDF-4 file (an HTML login or error page?)")
        part.rename(path)
    except BaseException:
        part.unlink(missing_ok=True)  # never leave a partial file behind
        raise
    return "downloaded"


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    urls = list_download_urls(SHORT_NAME, VERSION)
    print(f"{SHORT_NAME} {VERSION}: {len(urls)} files -> {OUTPUT_ROOT}")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, url): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            try:
                counts[future.result()] += 1
            except Exception as e:
                counts["failed"] += 1
                tqdm.write(f"Failed: {futures[future].rsplit('/', 1)[-1]}: {e}")

    print(", ".join(f"{k}: {v}" for k, v in counts.items()))
    if counts["failed"]:
        print("Re-run to retry the failed files.")


if __name__ == "__main__":
    main()
