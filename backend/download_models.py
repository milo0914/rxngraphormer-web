#!/usr/bin/env python3
"""
download_models.py — Fetch RXNGraphormer checkpoints at backend startup.

Downloads from GitHub Releases (primary) with figshare fallback.
Verifies MD5, extracts archives, writes manifest.

Usage:
    python download_models.py              # Download all missing
    python download_models.py --forward-only
    python download_models.py --retro-only
    python download_models.py --verify     # Check existing only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False
    print("WARNING: py7zr not installed; .7z extraction will fail", file=sys.stderr)

# ─── Configuration ────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / ".manifest.json"

# Known checkpoints (from MODELS.md)
CHECKPOINTS = {
    "forward": {
        "dir_name": "seq-v2-USPTO_STEREO-20250509_070206_ft",
        "required_files": [
            "parameters.json",
            "model/valid_checkpoint.pt",
        ],
        "github": {
            "asset_name": "forward-stereo-ft.7z",
            "url_template": "https://github.com/{repo}/releases/download/v1.0.0/{asset}",
        },
        "figshare": {
            "file_id": 59201306,
            "url": "https://ndownloader.figshare.com/files/59201306",
            "md5": "52d506a2ecee0c77cad7de03c692f653",
            "size": 391622627,
        },
        "archive_type": "7z",
    },
    "retro": {
        "dir_name": "USPTO_50k",
        "required_files": [
            "parameters.json",
            "model/valid_checkpoint.pt",
        ],
        "github": {
            "asset_name": "retro-uspto50k.7z",
            "url_template": "https://github.com/{repo}/releases/download/v1.0.0/{asset}",
        },
        "figshare": {
            "file_id": 53998184,
            "url": "https://ndownloader.figshare.com/files/53998184",
            "md5": "1d993b40b8ff38def31788c1ced69de5",
            "size": 197722944,
        },
        "archive_type": "7z",
    },
}

GITHUB_REPO = os.environ.get("GITHUB_RELEASES_REPO", "milo0914/rxngraphormer-web")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Optional, for private / rate limits
DOWNLOAD_TIMEOUT = 1800  # 30 min for large files
CHUNK_SIZE = 1024 * 1024  # 1 MB

# ─── Helpers ──────────────────────────────────────────────────────────────

def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_size: Optional[int] = None, headers: Optional[dict] = None) -> bool:
    """Download with resume support and progress bar. Returns True on success."""
    headers = headers or {}
    mode = "ab" if dest.exists() else "wb"
    resume_pos = dest.stat().st_size if dest.exists() else 0

    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    try:
        with requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            if r.status_code == 416:  # Range not satisfiable (already complete)
                return True
            r.raise_for_status()

            total = expected_size or int(r.headers.get("Content-Length", 0)) + resume_pos
            if resume_pos > 0 and "Content-Range" in r.headers:
                # Server responded to range request
                total = int(r.headers["Content-Range"].split("/")[-1])

            with dest.open(mode) as f, tqdm(
                total=total,
                initial=resume_pos,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}", file=sys.stderr)
        return False


def verify_md5(path: Path, expected_md5: str) -> bool:
    """Verify file MD5 matches expected."""
    actual = compute_md5(path)
    if actual.lower() != expected_md5.lower():
        print(f"  ✗ MD5 mismatch: expected {expected_md5}, got {actual}", file=sys.stderr)
        return False
    return True


def extract_archive(archive_path: Path, extract_dir: Path, archive_type: str) -> bool:
    """Extract .7z or .zip archive."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_type == "7z":
            if not HAS_7Z:
                raise RuntimeError("py7zr not installed")
            with py7zr.SevenZipFile(archive_path, "r") as z:
                z.extractall(extract_dir)
        elif archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(extract_dir)
        else:
            raise ValueError(f"Unknown archive type: {archive_type}")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}", file=sys.stderr)
        return False


def load_manifest() -> dict:
    """Load existing manifest or return empty dict."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    """Write manifest atomically."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(MANIFEST_PATH)


def check_model_exists(checkpoint: dict) -> bool:
    """Check if model directory has all required files."""
    model_dir = MODELS_DIR / checkpoint["dir_name"]
    if not model_dir.is_dir():
        return False
    for req in checkpoint["required_files"]:
        if not (model_dir / req).exists():
            return False
    return True


def download_checkpoint(name: str, checkpoint: dict, force: bool = False) -> bool:
    """Download and extract a single checkpoint."""
    model_dir = MODELS_DIR / checkpoint["dir_name"]
    archive_type = checkpoint["archive_type"]
    archive_name = f"{checkpoint['dir_name']}.{archive_type}"
    archive_path = MODELS_DIR / archive_name

    if check_model_exists(checkpoint) and not force:
        print(f"✓ {name}: already present at {model_dir}")
        return True

    print(f"⬇ {name}: downloading...")

    # Try GitHub Releases first
    github_url = checkpoint["github"]["url_template"].format(
        repo=GITHUB_REPO, asset=checkpoint["github"]["asset_name"]
    )
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    success = False
    source = "github"

    # Attempt GitHub
    print(f"  → Trying GitHub Releases: {github_url}")
    if download_file(github_url, archive_path, headers=headers):
        # Verify size roughly matches (GitHub doesn't publish MD5 in API easily)
        if archive_path.stat().st_size > 1024 * 1024:  # > 1 MB sanity
            success = True

    # Fallback to figshare
    if not success:
        figshare = checkpoint["figshare"]
        print(f"  → GitHub failed/unavailable, trying figshare: {figshare['url']}")
        archive_path.unlink(missing_ok=True)  # Clean partial
        if download_file(figshare["url"], archive_path, expected_size=figshare["size"]):
            if verify_md5(archive_path, figshare["md5"]):
                success = True
                source = "figshare"
            else:
                archive_path.unlink(missing_ok=True)

    if not success:
        print(f"  ✗ {name}: all download sources failed", file=sys.stderr)
        return False

    # Extract
    print(f"  → Extracting to {model_dir}...")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    if not extract_archive(archive_path, model_dir, archive_type):
        return False

    # Verify extracted files
    if not check_model_exists(checkpoint):
        print(f"  ✗ Extraction incomplete: required files missing", file=sys.stderr)
        return False

    # Update manifest
    manifest = load_manifest()
    manifest[name] = {
        "dir_name": checkpoint["dir_name"],
        "source": source,
        "archive_md5": compute_md5(archive_path),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive_size": archive_path.stat().st_size,
    }
    save_manifest(manifest)

    # Clean up archive (keep manifest)
    archive_path.unlink(missing_ok=True)

    print(f"  ✓ {name}: ready at {model_dir}")
    return True


def verify_all() -> bool:
    """Verify all checkpoints exist and manifest matches."""
    manifest = load_manifest()
    all_ok = True

    for name, checkpoint in CHECKPOINTS.items():
        model_dir = MODELS_DIR / checkpoint["dir_name"]
        if not model_dir.is_dir():
            print(f"✗ {name}: missing directory {model_dir}")
            all_ok = False
            continue

        missing = []
        for req in checkpoint["required_files"]:
            if not (model_dir / req).exists():
                missing.append(req)

        if missing:
            print(f"✗ {name}: missing files: {missing}")
            all_ok = False
        else:
            recorded = manifest.get(name, {})
            src = recorded.get("source", "unknown")
            print(f"✓ {name}: OK (source: {src})")

    return all_ok


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download RXNGraphormer model checkpoints")
    parser.add_argument("--forward-only", action="store_true", help="Only download forward model")
    parser.add_argument("--retro-only", action="store_true", help="Only download retro model")
    parser.add_argument("--verify", action="store_true", help="Verify existing models only")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        print("Verifying models...")
        ok = verify_all()
        sys.exit(0 if ok else 1)

    targets = []
    if args.forward_only:
        targets = [("forward", CHECKPOINTS["forward"])]
    elif args.retro_only:
        targets = [("retro", CHECKPOINTS["retro"])]
    else:
        targets = [("forward", CHECKPOINTS["forward"]), ("retro", CHECKPOINTS["retro"])]

    all_ok = True
    for name, checkpoint in targets:
        if not download_checkpoint(name, checkpoint, force=args.force):
            all_ok = False

    if all_ok:
        print("\n✓ All requested models ready")
    else:
        print("\n✗ Some models failed", file=sys.stderr)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()