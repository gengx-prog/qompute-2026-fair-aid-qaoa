"""Fetch the exact challenge instructions and primary references inside the workspace."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "downloads" / "track2" / "references"
SOURCES = {
    "challenge_README.md": "https://raw.githubusercontent.com/sandraalhames/Qompute-2026/855f3b7b7849b30cd63a07b730df97096fc8e317/README.md",
    "hadfield_alternating_operator.pdf": "https://arxiv.org/pdf/1709.03489",
    "ifrc_code_of_conduct.pdf": "https://www.ifrc.org/sites/default/files/2021-07/code-of-conduct-movement-ngos-english.pdf",
}


def download():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in SOURCES.items():
        request = urllib.request.Request(url, headers={"User-Agent": "Qompute-Track2-reference-check/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if filename.endswith(".pdf") and not content.startswith(b"%PDF"):
            raise ValueError(f"Expected PDF content for {filename}")
        path = DESTINATION / filename
        path.write_bytes(content)
        records.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), "url": url,
                        "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                        "downloaded_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Saved {path.name}: {len(content)} bytes", flush=True)
    (Path(__file__).resolve().parent / "reference_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    download()
