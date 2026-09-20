"""Download the age classifier weights into ./models and verify them against a pinned SHA-256.

The weights are NOT committed to this repository. They are Levi & Hassner's Adience age network
(Copyright 2015 Gil Levi and Tal Hassner, "please cite our paper"; no explicit open-source license),
converted to ONNX by the ONNX Model Zoo (Apache-2.0 repository). Read docs/MODEL_CARD.md before using or
redistributing them. Only the standard library is used so this can run before dependencies are installed.
"""

from __future__ import annotations

import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

ZOO_COMMIT = "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
AGE_MODEL = {
    "path": "age_googlenet.onnx",
    "url": f"https://github.com/onnx/models/raw/{ZOO_COMMIT}/validated/vision/body_analysis/age_gender/models/age_googlenet.onnx",
    "sha256": "fa2a3228e425056aa2b080b3afd3cf607327c86616e952602ed67b5fc16ab356",
    "size": 23_960_165,
}


def _ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle when installed (python.org builds on macOS ship without system certificates)."""
    try:
        import certifi  # noqa: PLC0415

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(dest_dir: Path, spec: dict[str, object] = AGE_MODEL) -> Path:
    dest = dest_dir / str(spec["path"])
    expected = str(spec["sha256"])
    if dest.is_file() and sha256_file(dest) == expected:
        print(f"{dest} already present and verified")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {spec['url']}")
    with urllib.request.urlopen(str(spec["url"]), timeout=60, context=_ssl_context()) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    actual = sha256_file(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch for {spec['url']}: expected {expected}, got {actual}")
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes, sha256 verified)")
    return dest


if __name__ == "__main__":
    fetch(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models"))
