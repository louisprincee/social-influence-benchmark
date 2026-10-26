"""Download the official Ciao MAT archive into the gitignored raw data directory."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw" / "ciao"
URL = "https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip"
EXPECTED_ZIP_SHA256 = "dee78c18b4a0843791e82b522f632a1166296e6048537b13b18c281a12e8f741"
EXPECTED_RATING_SHA256 = "aa793568686285a93286ba9e56954fcce347370ad2e58c785760362ab12d2c2c"
EXPECTED_TRUST_SHA256 = "0a8963ed22e6b534ce8d14e478f54ce0af6f5a3440a67ed5a35e62486580ce3a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ciao.zip"
    if not zip_path.exists():
        print(f"Downloading {URL}")
        urlretrieve(URL, zip_path)
    zip_hash = sha256_file(zip_path)
    if zip_hash != EXPECTED_ZIP_SHA256:
        raise ValueError(f"ciao.zip hash mismatch: {zip_hash}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DIR)
    rating = RAW_DIR / "ciao" / "rating.mat"
    trust = RAW_DIR / "ciao" / "trustnetwork.mat"
    if sha256_file(rating) != EXPECTED_RATING_SHA256:
        raise ValueError("rating.mat hash mismatch")
    if sha256_file(trust) != EXPECTED_TRUST_SHA256:
        raise ValueError("trustnetwork.mat hash mismatch")
    print(f"Verified Ciao MAT files in {RAW_DIR / 'ciao'}")


if __name__ == "__main__":
    main()
