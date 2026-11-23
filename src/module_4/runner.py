from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from .base import BaseModule4Detector, compute_suspicion_level, resolve_torch_device
from .forensic_translator import Module4ForensicTranslator

# Active Module 4 detectors.
# F3NetDetector remains in `src/module_4/f3net_detector.py` (like ELA in Module 3)
# and is intentionally omitted from the active registry until re-enabled.
DETECTOR_REGISTRY: Dict[str, Type[BaseModule4Detector]] = {}


def register_detector(detector_cls: Type[BaseModule4Detector]) -> None:
    if not detector_cls.detector_id:
        raise ValueError("Detector class must define a non-empty `detector_id`.")
    DETECTOR_REGISTRY[detector_cls.detector_id] = detector_cls


def list_available_detectors() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for det_id, cls in DETECTOR_REGISTRY.items():
        inst = cls()
        items.append(
            {
                "detector_id": det_id,
                "display_name": inst.display_name,
                "paradigm": inst.paradigm,
                "input_size": inst.input_size,
                "crop_spec": cls.get_crop_spec(),
                "weights_cached": inst.has_local_weights(),
            }
        )
    return items


def get_required_crop_specs(
    detector_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Collect unique Module 2 CropSpecs required by active Module 4 detectors
    without instantiating models or downloading weights.
    """
    target_ids = (
        list(detector_ids)
        if detector_ids is not None
        else list(DETECTOR_REGISTRY.keys())
    )
    unique_specs: Dict[str, Dict[str, Any]] = {}
    for det_id in target_ids:
        cls = DETECTOR_REGISTRY.get(det_id)
        if cls is None:
            continue
        spec = cls.get_crop_spec()
        if spec and spec.get("spec_id") and spec["spec_id"] not in unique_specs:
            unique_specs[spec["spec_id"]] = spec
    return list(unique_specs.values())


def _get_manifest_dir(m2_manifest: Dict[str, Any]) -> Path:
    artifacts_dir = m2_manifest.get("artifacts_directory", {})
    manifest_ref = m2_manifest.get("manifest_path") or artifacts_dir.get("manifest")
    if manifest_ref:
        return Path(manifest_ref).parent
    case_id = m2_manifest.get("forensics_metadata", {}).get("case_id", "case")
    return Path(artifacts_dir.get("root", Path("data/processed") / case_id))


def run_module4(
    m2_manifest: Dict[str, Any],
    detector_ids: Optional[Sequence[str]] = None,
    device: str = "auto",
    auto_download: bool = True,
    batch_size: int = 4,
    enable_gradcam: bool = True,
) -> Dict[str, Any]:
    """
    Execute Module 4 detectors using Sequential Lazy-Loading:
      Loads one detector -> evaluates pre-cropped inputs from Module 2 ->
      saves XAI overlays -> unloads model and frees RAM/VRAM before loading the next.
    """
    manifest_dir = _get_manifest_dir(m2_manifest)
    m4_root = manifest_dir / "module4"
    m4_root.mkdir(parents=True, exist_ok=True)

    target_ids = (
        list(detector_ids)
        if detector_ids is not None
        else list(DETECTOR_REGISTRY.keys())
    )
    resolved_device = str(resolve_torch_device(device))

    detector_outputs: Dict[str, Dict[str, Any]] = {}
    detector_summaries: Dict[str, Dict[str, Any]] = {}
    detectors_executed: List[str] = []
    detectors_skipped: List[Dict[str, str]] = []

    for det_id in target_ids:
        cls = DETECTOR_REGISTRY.get(det_id)
        if cls is None:
            detectors_skipped.append(
                {"detector_id": det_id, "reason": "Not registered in DETECTOR_REGISTRY"}
            )
            continue

        detector = cls()
        det_out_dir = m4_root / det_id
        try:
            detector.load(device=resolved_device, auto_download=auto_download)
            res = detector.analyze_case(
                m2_manifest=m2_manifest,
                output_dir=str(det_out_dir),
                batch_size=batch_size,
                enable_gradcam=enable_gradcam,
            )
            detector_outputs[det_id] = res
            detector_summaries[det_id] = {
                "detector_id": res["detector_id"],
                "display_name": res["display_name"],
                "paradigm": res["paradigm"],
                "input_size": res["input_size"],
                "crop_spec": cls.get_crop_spec(),
                "weights_path": res["weights_path"],
                "device_used": res["device_used"],
                **res["summary"],
            }
            detectors_executed.append(det_id)
        except Exception as exc:
            detectors_skipped.append({"detector_id": det_id, "reason": str(exc)})
        finally:
            detector.unload()

    # Build frame_index lookup maps to avoid positional indexing assumptions
    det_frame_lookup: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for det_id in detectors_executed:
        by_idx: Dict[int, Dict[str, Any]] = {}
        for fr in detector_outputs[det_id].get("frames", []) or []:
            if fr.get("frame_index") is not None:
                by_idx[int(fr["frame_index"])] = fr
        det_frame_lookup[det_id] = by_idx

    frames_detail_m2 = m2_manifest.get("frames_detail", []) or []
    translator = Module4ForensicTranslator()
    combined_frames: List[Dict[str, Any]] = []
    frame_scores: List[float] = []
    used_names: set[str] = set()

    for idx, frame_rec in enumerate(frames_detail_m2):
        frame_idx = int(frame_rec.get("frame_display_index", idx + 1))
        frame_path = frame_rec.get("frame_path", "")
        frame_stem = Path(frame_path).stem if frame_path else f"frame_{frame_idx:05d}"

        per_detector_frame: Dict[str, Dict[str, Any]] = {}
        per_detector_interp: Dict[str, Dict[str, Any]] = {}
        active_probs: List[float] = []

        for det_id in detectors_executed:
            det_res = detector_outputs[det_id]
            det_frame = det_frame_lookup[det_id].get(frame_idx)
            if det_frame is None:
                det_frame = {
                    "frame_index": frame_idx,
                    "frame_path": frame_path,
                    "evaluated": False,
                    "fake_probability": None,
                    "suspicion_level": "low",
                    "peak_face_id": None,
                    "faces": [],
                }
            per_detector_frame[det_id] = det_frame
            per_detector_interp[det_id] = translator.interpret_detector_frame(
                detector_id=det_id,
                display_name=det_res["display_name"],
                frame_rec=det_frame,
            )
            if det_frame.get("evaluated") and det_frame.get("fake_probability") is not None:
                active_probs.append(float(det_frame["fake_probability"]))

        evaluated = len(active_probs) > 0
        if evaluated:
            ensemble_score = round(float(np.mean(active_probs)), 4)
            frame_scores.append(ensemble_score)
        else:
            ensemble_score = 0.0

        rec_name = f"{frame_stem}_module4_record.json"
        if rec_name in used_names:
            rec_name = f"{frame_stem}_{idx + 1:05d}_module4_record.json"
        used_names.add(rec_name)
        rec_path = manifest_dir / rec_name

        frame_m4_record = {
            "frame_index": frame_idx,
            "frame_path": frame_path,
            "evaluated": evaluated,
            "ensemble_fake_probability": ensemble_score if evaluated else None,
            "suspicion_level": compute_suspicion_level(ensemble_score),
            "detectors": per_detector_frame,
            "interpretation": per_detector_interp,
            "record_json_path": str(rec_path.resolve()),
        }
        rec_path.write_text(
            json.dumps(frame_m4_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        combined_frames.append(frame_m4_record)

    if frame_scores:
        mean_m4 = round(float(np.mean(frame_scores)), 4)
        max_m4 = round(float(np.max(frame_scores)), 4)
        sorted_desc = sorted(frame_scores, reverse=True)
        k = max(1, math.ceil(0.30 * len(sorted_desc)))
        topk_m4 = round(float(np.mean(sorted_desc[:k])), 4)
    else:
        mean_m4 = 0.0
        max_m4 = 0.0
        topk_m4 = 0.0

    # Video-level aggregation uses mean probability across evaluated frames to avoid
    # single-frame blur/occlusion outliers triggering false HIGH alerts.
    is_multi_frame = len(combined_frames) > 1
    primary_case_score = mean_m4 if is_multi_frame else max_m4

    m4_manifest_path = manifest_dir / "module4_manifest.json"
    m4_manifest: Dict[str, Any] = {
        "forensics_metadata": {
            "case_id": m2_manifest["forensics_metadata"]["case_id"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "media_type": m2_manifest["forensics_metadata"]["media_type"],
        },
        "module4_settings": {
            "execution_mode": "Sequential Lazy-Loading",
            "device_target": resolved_device,
            "gradcam_enabled": bool(enable_gradcam),
            "batch_size": int(batch_size),
            "case_aggregation": "mean" if is_multi_frame else "single_frame",
        },
        "detectors_executed": detectors_executed,
        "detectors_skipped": detectors_skipped,
        "detector_summaries": detector_summaries,
        "execution_summary": {
            "frames_total": len(combined_frames),
            "frames_evaluated": len(frame_scores),
            "mean_module4_score": mean_m4,
            "max_module4_score": max_m4,
            "topk_module4_score": topk_m4,
            "primary_module4_score": primary_case_score,
            "overall_suspicion": compute_suspicion_level(primary_case_score),
        },
        "artifacts_directory": {
            "root": str(m4_root.resolve()),
            "manifest": str(m4_manifest_path.resolve()),
        },
        "frames_detail": combined_frames,
    }
    m4_manifest["interpretation"] = translator.interpret_manifest(m4_manifest)

    m4_manifest_path.write_text(
        json.dumps(m4_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return m4_manifest
