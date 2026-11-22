from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from .face_aligner import FaceAligner, _imread_unicode, _is_valid_face_geometry
from .frame_parser import FrameParser


def _detect_lightweight_faces(
    aligner: FaceAligner, frame_path: str
) -> List[Dict[str, Any]]:
    """
    Detect bounding boxes and 5-point landmarks on a consecutive burst frame
    without writing extra aligned crops or annotated overlays to disk.
    """
    img_bgr = _imread_unicode(Path(frame_path))
    if img_bgr is None:
        return []

    detections = aligner._detect(img_bgr)
    faces: List[Dict[str, Any]] = []
    valid_idx = 0
    for box, prob, pts_arr in detections:
        if not _is_valid_face_geometry(box, pts_arr, aligner.min_face_size):
            continue
        x1, y1, x2, y2 = [round(float(v), 2) for v in box]
        faces.append(
            {
                "face_index": valid_idx,
                "confidence": round(float(prob), 4),
                "bbox": [x1, y1, x2, y2],
                "landmarks": {
                    "left_eye": [round(float(pts_arr[0][0]), 2), round(float(pts_arr[0][1]), 2)],
                    "right_eye": [round(float(pts_arr[1][0]), 2), round(float(pts_arr[1][1]), 2)],
                    "nose": [round(float(pts_arr[2][0]), 2), round(float(pts_arr[2][1]), 2)],
                    "mouth_left": [round(float(pts_arr[3][0]), 2), round(float(pts_arr[3][1]), 2)],
                    "mouth_right": [round(float(pts_arr[4][0]), 2), round(float(pts_arr[4][1]), 2)],
                },
            }
        )
        valid_idx += 1
    return faces


def run_module2(
    input_path: str,
    processed_root: str = "data/processed",
    case_id: Optional[str] = None,
    output_size: Tuple[int, int] = (299, 299),
    max_frames: Optional[int] = 10,
    min_confidence: float = 0.80,
    backend: str = "retinaface",
    burst_length: int = 5,
) -> Dict[str, Any]:
    """
    Pipeline Module 2: Preprocessing & ROI Extraction
    - Extracts evenly spaced non-blank moments (exact requested count when available).
    - Extracts short consecutive frame bursts for video temporal analysis.
    - Aligns faces on primary moments and records lightweight landmarks on burst frames.
    """
    src = Path(input_path)
    folder_name = case_id or src.stem
    case_dir = Path(processed_root) / folder_name
    frames_dir = case_dir / "frames"
    aligned_dir = case_dir / "aligned_faces"
    annotated_dir = case_dir / "annotated"

    for sub_dir in (frames_dir, aligned_dir, annotated_dir):
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

    # 1. Frame & Burst Parsing
    parser = FrameParser(max_frames=max_frames, burst_length=burst_length)
    parse_result = parser.parse(str(src), str(frames_dir))

    # 2. Face Alignment on primary frames + lightweight tracking on burst frames
    aligner = FaceAligner(
        backend=backend,
        output_size=output_size,
        min_confidence=min_confidence,
    )

    frame_reports: List[Dict[str, Any]] = []
    all_aligned_faces: List[str] = []
    face_global_counter = 0

    bursts_per_anchor = parse_result.get("bursts") or [
        [fp] for fp in parse_result["frames"]
    ]

    for frame_idx, (frame_path, burst_paths) in enumerate(
        zip(parse_result["frames"], bursts_per_anchor)
    ):
        report = aligner.process_frame(
            frame_path=frame_path,
            aligned_dir=str(aligned_dir),
            annotated_dir=str(annotated_dir),
        )
        report["frame_display_index"] = frame_idx + 1

        for face in report["faces"]:
            face["global_face_id"] = f"Face_{face_global_counter:03d}"
            all_aligned_faces.append(face["aligned_face_path"])
            face_global_counter += 1

        # Collect lightweight face coordinates on consecutive burst frames for Module 3 Temporal check
        burst_records: List[Dict[str, Any]] = []
        for b_idx, b_path in enumerate(burst_paths):
            if Path(b_path).name.endswith("_b00.png") or b_path == frame_path:
                b_faces = [
                    {
                        "face_index": f["face_index"],
                        "confidence": f["confidence"],
                        "bbox": f["bbox"],
                        "landmarks": f["landmarks"],
                    }
                    for f in report["faces"]
                ]
            else:
                b_faces = _detect_lightweight_faces(aligner, b_path)

            burst_records.append(
                {
                    "burst_index": b_idx,
                    "frame_path": b_path,
                    "faces": b_faces,
                }
            )

        report["burst_frames"] = burst_records
        frame_reports.append(report)

    # 3. Manifest Output
    manifest_data: Dict[str, Any] = {
        "forensics_metadata": {
            "module": "Module 2: Preprocessing & ROI Extraction",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": folder_name,
            "source_file": src.name,
            "source_path": str(src.resolve()),
            "media_type": parse_result["media_type"].upper(),
            "fps": parse_result.get("fps"),
        },
        "preprocessing_settings": {
            "ai_detector": backend.upper(),
            "confidence_threshold": min_confidence,
            "canonical_resolution": f"{output_size[0]}x{output_size[1]}",
            "sampling_mode": parse_result.get("sampling_mode", "Standard"),
            "target_frames_limit": max_frames,
            "burst_length": parse_result.get("burst_length", 1),
        },
        "execution_summary": {
            "total_frames_inspected": parse_result.get(
                "total_raw_keyframes", len(parse_result["frames"])
            ),
            "discarded_blank_or_extreme_frames": parse_result.get(
                "discarded_blank_frames", 0
            ),
            "final_frames_analyzed": len(frame_reports),
            "total_faces_extracted": len(all_aligned_faces),
            "average_faces_per_frame": round(
                len(all_aligned_faces) / max(len(frame_reports), 1), 2
            ),
        },
        "artifacts_directory": {
            "root": str(case_dir.resolve()),
            "frames_folder": str(frames_dir.resolve()),
            "aligned_faces_folder": str(aligned_dir.resolve()),
            "annotated_folder": str(annotated_dir.resolve()),
        },
        "aligned_face_paths": all_aligned_faces,
        "frames_detail": frame_reports,
    }

    manifest_path = case_dir / "preprocessing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    manifest_data["manifest_path"] = str(manifest_path.resolve())
    return manifest_data


__all__ = ["FrameParser", "FaceAligner", "run_module2"]
