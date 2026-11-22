from __future__ import annotations

from abc import ABC, abstractmethod
import gc
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .weight_store import ensure_model_weights, find_existing_weights


def compute_suspicion_level(score: float) -> str:
    """Unified Module 3 & Module 4 severity thresholds."""
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def resolve_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class BaseModule4Detector(ABC):
    """
    Shared plugin contract for all Module 4 Deep Learning & XAI detectors:
      - F3Net (DeepfakeBench FAD-DCT)
      - Xception (DeepfakeBench Spatial CNN)
      - SBI / EfficientNet-B4 (Blending Generalization)
      - Community Forensics ViT-S/16 (Modern Generators)

    Designed for Sequential Lazy-Loading on CPU / 8GB RAM systems:
      load() -> analyze_case() -> unload()
    """

    detector_id: str = ""
    display_name: str = ""
    paradigm: str = ""
    input_size: int = 256
    input_source: str = "aligned_face"  # "aligned_face" | "raw_bbox" | "raw_frame"
    requires_face: bool = True

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.custom_weights_path = weights_path
        self.device: torch.device = torch.device("cpu")
        self._is_loaded: bool = False

    def resolve_weights_path(self, auto_download: bool = True) -> str:
        if self.custom_weights_path:
            p = Path(self.custom_weights_path)
            if not p.exists():
                raise FileNotFoundError(f"Custom weights not found: {p}")
            return str(p.resolve())
        return ensure_model_weights(self.detector_id, auto_download=auto_download)

    def has_local_weights(self) -> bool:
        if self.custom_weights_path:
            return Path(self.custom_weights_path).exists()
        return find_existing_weights(self.detector_id) is not None

    @abstractmethod
    def load(self, device: str = "auto", auto_download: bool = True) -> None:
        """Load checkpoint and model onto target device."""

    @abstractmethod
    def analyze_case(
        self,
        m2_manifest: Dict[str, Any],
        output_dir: str,
        batch_size: int = 4,
        enable_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference and XAI (Grad-CAM) across all relevant frames/faces in Module 2's manifest.
        Must return a standardized dict with:
          - detector_id, display_name, paradigm, input_size, weights_path
          - frames: list of per-frame dicts keyed by frame_index
          - summary: case-level scores (mean_fake_probability, max_fake_probability,
                     topk_fake_probability, suspicion_level)
        """

    def unload(self) -> None:
        """Release model memory and GPU cache for Sequential Lazy-Loading."""
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
