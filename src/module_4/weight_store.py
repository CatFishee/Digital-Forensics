from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Optional, Tuple

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "~/.digital_forensics/models/module_4"
_ENV_KEY = "DIGITAL_FORENSICS_CACHE_DIR"
_LOCAL_FALLBACK_DIR = Path("weights/module_4")
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ModelWeightSpec:
    model_id: str
    filename: str
    urls: Tuple[str, ...]
    sha256: Optional[str] = None
    sha256_prefix: Optional[str] = None
    license_tag: str = "Unknown"
    source_note: str = ""


# Registry of Module 4 checkpoints.
# Classmates can add entries for "xception", "sbi", and "commfor" here.
MODEL_REGISTRY: Dict[str, ModelWeightSpec] = {
    "f3net": ModelWeightSpec(
        model_id="f3net",
        filename="f3net_best.pth",
        urls=(
            "https://huggingface.co/ADPer/verity-weights/resolve/main/deepfakebench/f3net_best.pth",
        ),
        sha256=None,
        sha256_prefix="deee7d25bfc1b8b5",
        license_tag="CC-BY-NC-4.0",
        source_note="DeepfakeBench F3Net (FAD branch, FaceForensics++ c23)",
    ),
}


def get_cache_dir() -> str:
    """
    Resolve Module 4 cache directory in the same spirit as uniface.model_store:
      1. DIGITAL_FORENSICS_CACHE_DIR env var (if set)
      2. Default: ~/.digital_forensics/models/module_4
    """
    return os.path.expanduser(os.environ.get(_ENV_KEY, _DEFAULT_CACHE_DIR))


def set_cache_dir(path: str) -> None:
    """Override the Module 4 model cache directory via environment variable."""
    os.environ[_ENV_KEY] = path
    logger.info("Module 4 cache directory set to: %s", path)


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(
    file_path: str,
    expected_sha256: Optional[str] = None,
    expected_prefix: Optional[str] = None,
) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False

    actual = compute_sha256(file_path)
    if expected_sha256:
        if actual.lower() != expected_sha256.lower():
            logger.warning(
                "SHA-256 mismatch for %s: expected %s, got %s",
                file_path,
                expected_sha256,
                actual,
            )
            return False
        return True

    if expected_prefix:
        if not actual.lower().startswith(expected_prefix.lower()):
            logger.warning(
                "SHA-256 prefix mismatch for %s: expected prefix %s, got %s",
                file_path,
                expected_prefix,
                actual,
            )
            return False
        return True

    return True


def _download_file(
    url: str,
    dest_path: str,
    expected_sha256: Optional[str] = None,
    expected_prefix: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> None:
    last_error: Optional[Exception] = None
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    for attempt in range(max_retries):
        tmp_path: Optional[str] = None
        try:
            headers = {"User-Agent": "Digital-Forensics-Toolkit/1.0"}
            with requests.get(
                url, stream=True, timeout=timeout, headers=headers, allow_redirects=True
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
                with (
                    os.fdopen(fd, "wb") as f,
                    tqdm(
                        total=total_size,
                        desc=f"Downloading {Path(dest_path).name} ({attempt + 1}/{max_retries})",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress,
                ):
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))

            if not verify_file_hash(tmp_path, expected_sha256, expected_prefix):
                raise ValueError("SHA-256 verification failed on downloaded checkpoint")

            os.replace(tmp_path, dest_path)
            return
        except (OSError, requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    raise ConnectionError(
        f"Failed to download checkpoint from {url} after {max_retries} attempts: {last_error}"
    )


def find_existing_weights(model_id: str, root: Optional[str] = None) -> Optional[str]:
    """
    Return the local checkpoint path if already cached (without triggering a download).
    Checks:
      1. Cache dir (~/.digital_forensics/models/module_4/<filename> or DIGITAL_FORENSICS_CACHE_DIR)
      2. Project-local fallback (weights/module_4/<filename>)
    """
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown Module 4 model_id: '{model_id}'")

    cache_root = Path(os.path.expanduser(root) if root is not None else get_cache_dir())
    candidates = [
        cache_root / spec.filename,
        _LOCAL_FALLBACK_DIR / spec.filename,
    ]
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return str(cand.resolve())
    return None


def ensure_model_weights(
    model_id: str,
    root: Optional[str] = None,
    auto_download: bool = True,
    timeout: int = 120,
    max_retries: int = 3,
) -> str:
    """
    Ensure weights for `model_id` exist locally and pass SHA-256 verification.
    Downloads atomically on first use if `auto_download=True`.
    """
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown Module 4 model_id: '{model_id}'")

    existing = find_existing_weights(model_id, root=root)
    if existing:
        if verify_file_hash(existing, spec.sha256, spec.sha256_prefix):
            return existing
        logger.warning("Corrupted weights detected at %s; removing.", existing)
        os.remove(existing)

    if not auto_download:
        cache_root = os.path.expanduser(root) if root is not None else get_cache_dir()
        expected_loc = os.path.join(cache_root, spec.filename)
        raise FileNotFoundError(
            f"Weights for '{model_id}' not found at '{expected_loc}' and auto_download=False."
        )

    if not spec.urls:
        raise FileNotFoundError(
            f"No download URL registered for '{model_id}'. Please place '{spec.filename}' in '{get_cache_dir()}'."
        )

    cache_root = os.path.expanduser(root) if root is not None else get_cache_dir()
    os.makedirs(cache_root, exist_ok=True)
    dest_path = os.path.normpath(os.path.join(cache_root, spec.filename))

    last_err: Optional[Exception] = None
    for url in spec.urls:
        try:
            _download_file(
                url=url,
                dest_path=dest_path,
                expected_sha256=spec.sha256,
                expected_prefix=spec.sha256_prefix,
                timeout=timeout,
                max_retries=max_retries,
            )
            return str(Path(dest_path).resolve())
        except Exception as exc:
            last_err = exc
            logger.warning("Failed downloading %s from %s: %s", model_id, url, exc)

    raise ConnectionError(
        f"Unable to download weights for '{model_id}' from any registered source: {last_err}"
    )
