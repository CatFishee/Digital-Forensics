from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


@dataclass(frozen=True)
class CropSpec:
    """
    Declarative crop specification requested by a Module 4 detector so Module 2
    exports model-native single-pass crops directly from raw frames.
    """

    spec_id: str
    crop_mode: str = "aligned_square"  # "aligned_square" | "aligned_bbox_margin" | "raw_bbox" | "raw_frame"
    output_size: Tuple[int, int] = (256, 256)
    margin_scale: float = 0.85
    bbox_margin: float = 1.3
    border_mode: str = "replicate"  # "replicate" | "constant_black" | "constant_clip_mean" | "reflect101"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "crop_mode": self.crop_mode,
            "output_size": [int(self.output_size[0]), int(self.output_size[1])],
            "margin_scale": float(self.margin_scale),
            "bbox_margin": float(self.bbox_margin),
            "border_mode": self.border_mode,
        }


class BaseModule4Detector(ABC):
    """
    Shared plugin contract for all Module 4 Deep Learning & XAI detectors.

    Designed for Sequential Lazy-Loading on CPU / 8GB RAM systems:
      get_crop_spec() -> [Module 2 exports crops] -> load() -> analyze_case() -> unload()
    """

    detector_id: str = ""
    display_name: str = ""
    paradigm: str = ""
    input_size: int = 256
    input_source: str = "aligned_face"  # "aligned_face" | "raw_bbox" | "raw_frame"
    requires_face: bool = True
    crop_spec: Optional[CropSpec] = None

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.custom_weights_path = weights_path
        self.device: torch.device = torch.device("cpu")
        self._is_loaded: bool = False

    @classmethod
    def get_crop_spec(cls) -> Optional[Dict[str, Any]]:
        """
        Return the detector's Module 2 CropSpec dictionary without loading model weights.
        If `crop_spec` is not overridden, defaults to a single-pass aligned square crop
        at `cls.input_size`.
        """
        if cls.crop_spec is not None:
            return cls.crop_spec.to_dict()
        if not cls.requires_face or cls.input_source == "raw_frame":
            return None
        mode = "raw_bbox" if cls.input_source == "raw_bbox" else "aligned_square"
        return CropSpec(
            spec_id=f"{cls.detector_id or 'default'}_{cls.input_size}",
            crop_mode=mode,
            output_size=(cls.input_size, cls.input_size),
        ).to_dict()

    def resolve_face_crop_path(self, face_info: Dict[str, Any]) -> Optional[str]:
        """
        Resolve the detector's dedicated Module 2 crop path from `face_info["model_crops"]`,
        falling back to `face_info["aligned_face_path"]` for backward compatibility.
        """
        spec = self.get_crop_spec()
        if spec:
            crop_entry = (face_info.get("model_crops") or {}).get(spec["spec_id"])
            if crop_entry and crop_entry.get("path"):
                p = Path(crop_entry["path"])
                if p.exists():
                    return str(p)
        fallback = face_info.get("aligned_face_path")
        if fallback and Path(fallback).exists():
            return str(fallback)
        return None

    def get_face_crop_meta(self, face_info: Dict[str, Any]) -> Dict[str, Any]:
        """Return crop metadata (`padded_ratio`, `unreliable_pose_warning`) for the active spec."""
        spec = self.get_crop_spec()
        if spec:
            crop_entry = (face_info.get("model_crops") or {}).get(spec["spec_id"])
            if crop_entry:
                return {
                    "padded_ratio": float(crop_entry.get("padded_ratio", 0.0)),
                    "unreliable_pose_warning": bool(
                        crop_entry.get("unreliable_pose_warning", False)
                    ),
                    "output_size": crop_entry.get("output_size", spec["output_size"]),
                    "crop_mode": crop_entry.get("crop_mode", spec["crop_mode"]),
                }
        return {
            "padded_ratio": float(face_info.get("padded_ratio", 0.0)),
            "unreliable_pose_warning": bool(
                face_info.get("unreliable_pose_warning", False)
            ),
            "output_size": face_info.get("aligned_size", [self.input_size, self.input_size]),
            "crop_mode": "aligned_square",
        }

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
