"""Download the official portable Tectonic release into this paper's tools folder."""
from pathlib import Path
import hashlib
import io
import json
import requests
import zipfile

destination = Path(__file__).resolve().parent / "tools"
destination.mkdir(exist_ok=True)
response = requests.get("https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest", timeout=45)
response.raise_for_status()
release = response.json()
asset = next(a for a in release["assets"] if a["name"].endswith("x86_64-pc-windows-msvc.zip"))
response = requests.get(asset["browser_download_url"], timeout=120)
response.raise_for_status()
payload = response.content
actual_digest = hashlib.sha256(payload).hexdigest()
expected_digest = asset.get("digest")
if expected_digest and expected_digest.startswith("sha256:"):
    assert expected_digest == "sha256:" + actual_digest, "Release asset checksum mismatch"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    executable = next(n for n in archive.namelist() if Path(n).name == "tectonic.exe")
    (destination / "tectonic.exe").write_bytes(archive.read(executable))
(destination / "download_provenance.json").write_text(json.dumps({"release": release["tag_name"], "url": asset["browser_download_url"], "sha256": actual_digest, "github_digest": expected_digest}, indent=2), encoding="utf-8")
print(f"Downloaded {asset['name']} ({len(payload):,} bytes) from the official Tectonic repository.")
