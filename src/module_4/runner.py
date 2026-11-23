from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from .base import BaseModule4Detector, compute_suspicion_level, resolve_torch_device
from .community_vit import CommunityForensicsDetector
from .forensic_translator import Module4ForensicTranslator

# Active Module 4 detectors.
DETECTOR_REGISTRY: Dict[str, Type[BaseModule4Detector]] = {
    CommunityForensicsDetector.detector_id: CommunityForensicsDetector,
}


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


def _build_detector_summary(
    cls: Type[BaseModule4Detector], res: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "detector_id": res["detector_id"],
        "display_name": res["display_name"],
        "paradigm": res["paradigm"],
        "input_size": res["input_size"],
        "crop_spec": cls.get_crop_spec(),
        "weights_path": res["weights_path"],
        "device_used": res["device_used"],
        **res["summary"],
    }


def _run_sequential(
    target_ids: List[str],
    m2_manifest: Dict[str, Any],
    m4_root: Path,
    resolved_device: str,
    auto_download: bool,
    batch_size: int,
    enable_gradcam: bool,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[str],
    List[Dict[str, str]],
]:
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
            detector_summaries[det_id] = _build_detector_summary(cls, res)
            detectors_executed.append(det_id)
        except Exception as exc:
            detectors_skipped.append({"detector_id": det_id, "reason": str(exc)})
        finally:
            detector.unload()

    return detector_outputs, detector_summaries, detectors_executed, detectors_skipped


def _run_parallel(
    target_ids: List[str],
    m2_manifest: Dict[str, Any],
    m4_root: Path,
    resolved_device: str,
    auto_download: bool,
    batch_size: int,
    enable_gradcam: bool,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[str],
    List[Dict[str, str]],
]:
    """
    Parallel Execution Mode (for high-VRAM GPUs):
      1. Safely pre-load all selected detectors into VRAM/RAM.
      2. Run `analyze_case` concurrently across threads (sharing 1 CUDA context).
      3. Unload all models in finally block.
    """
    detector_outputs: Dict[str, Dict[str, Any]] = {}
    detector_summaries: Dict[str, Dict[str, Any]] = {}
    detectors_executed: List[str] = []
    detectors_skipped: List[Dict[str, str]] = []

    loaded_items: List[Tuple[str, Type[BaseModule4Detector], BaseModule4Detector]] = []
    try:
        # 1. Pre-load an toàn (tránh xung đột cache/download khi nhiều thread cùng tải)
        for det_id in target_ids:
            cls = DETECTOR_REGISTRY.get(det_id)
            if cls is None:
                detectors_skipped.append(
                    {"detector_id": det_id, "reason": "Not registered in DETECTOR_REGISTRY"}
                )
                continue

            detector = cls()
            try:
                detector.load(device=resolved_device, auto_download=auto_download)
                loaded_items.append((det_id, cls, detector))
            except Exception as exc:
                detector.unload()
                detectors_skipped.append({"detector_id": det_id, "reason": str(exc)})

        # 2. Chạy suy luận & Grad-CAM song song
        if loaded_items:
            def _worker(
                item: Tuple[str, Type[BaseModule4Detector], BaseModule4Detector]
            ) -> Tuple[str, Type[BaseModule4Detector], Dict[str, Any]]:
                d_id, d_cls, d_inst = item
                res = d_inst.analyze_case(
                    m2_manifest=m2_manifest,
                    output_dir=str(m4_root / d_id),
                    batch_size=batch_size,
                    enable_gradcam=enable_gradcam,
                )
                return d_id, d_cls, res

            temp_results: Dict[str, Tuple[Type[BaseModule4Detector], Dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=len(loaded_items)) as executor:
                future_map = {
                    executor.submit(_worker, item): item[0] for item in loaded_items
                }
                for fut in as_completed(future_map):
                    d_id = future_map[fut]
                    try:
                        res_id, res_cls, res_data = fut.result()
                        temp_results[res_id] = (res_cls, res_data)
                    except Exception as exc:
                        detectors_skipped.append({"detector_id": d_id, "reason": str(exc)})

            # Giữ đúng thứ tự ưu tiên như target_ids
            for det_id in target_ids:
                if det_id in temp_results:
                    d_cls, d_res = temp_results[det_id]
                    detector_outputs[det_id] = d_res
                    detector_summaries[det_id] = _build_detector_summary(d_cls, d_res)
                    detectors_executed.append(det_id)
    finally:
        for _, _, detector in loaded_items:
            try:
                detector.unload()
            except Exception:
                pass

    return detector_outputs, detector_summaries, detectors_executed, detectors_skipped


def run_module4(
    m2_manifest: Dict[str, Any],
    detector_ids: Optional[Sequence[str]] = None,
    device: str = "auto",
    auto_download: bool = True,
    batch_size: int = 4,
    enable_gradcam: bool = True,
    execution_mode: str = "sequential",  # "sequential" | "parallel"
) -> Dict[str, Any]:
    """
    Execute Module 4 detectors using either:
      - "sequential": Sequential Lazy-Loading (low RAM / 8GB machines).
      - "parallel": Concurrent GPU/Multi-thread execution (high-VRAM GPUs).
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
    mode_norm = "parallel" if str(execution_mode).lower() == "parallel" else "sequential"

    if mode_norm == "parallel" and len(target_ids) > 1:
        (
            detector_outputs,
            detector_summaries,
            detectors_executed,
            detectors_skipped,
        ) = _run_parallel(
            target_ids=target_ids,
            m2_manifest=m2_manifest,
            m4_root=m4_root,
            resolved_device=resolved_device,
            auto_download=auto_download,
            batch_size=batch_size,
            enable_gradcam=enable_gradcam,
        )
        mode_display = "Parallel Concurrent (Pre-loaded)"
    else:
        (
            detector_outputs,
            detector_summaries,
            detectors_executed,
            detectors_skipped,
        ) = _run_sequential(
            target_ids=target_ids,
            m2_manifest=m2_manifest,
            m4_root=m4_root,
            resolved_device=resolved_device,
            auto_download=auto_download,
            batch_size=batch_size,
            enable_gradcam=enable_gradcam,
        )
        mode_display = "Sequential Lazy-Loading"

    # Build frame_index lookup maps
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

    is_multi_frame = len(combined_frames) > 1
    primary_case_score = topk_m4 if is_multi_frame else max_m4

    m4_manifest_path = manifest_dir / "module4_manifest.json"
    m4_manifest: Dict[str, Any] = {
        "forensics_metadata": {
            "case_id": m2_manifest["forensics_metadata"]["case_id"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "media_type": m2_manifest["forensics_metadata"]["media_type"],
        },
        "module4_settings": {
            "execution_mode": mode_display,
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