import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from .face_aligner import FaceAligner
from .frame_parser import FrameParser
import shutil


def run_module2(
    input_path: str,
    processed_root: str = "data/processed",
    case_id: Optional[str] = None,
    output_size: Tuple[int, int] = (299, 299),
    max_frames: Optional[int] = 30,
    min_confidence: float = 0.80,
    backend: str = "retinaface",
) -> Dict[str, Any]:
    """
    Pipeline đầy đủ cho Module 2 (Preprocessing & ROI Extraction):
    1. Frame Parsing (Ảnh giữ nguyên / Video trích xuất I-frames).
    2. Face Localization & Landmarks (RetinaFace / MTCNN).
    3. Affine Transformation về Canonical Pose & lưu manifest JSON.
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

    parser = FrameParser(max_frames=max_frames)
    parse_result = parser.parse(str(src), str(frames_dir))

    aligner = FaceAligner(
        backend=backend,
        output_size=output_size,
        min_confidence=min_confidence,
    )

    frame_reports = []
    all_aligned_faces = []
    for frame_path in parse_result["frames"]:
        report = aligner.process_frame(
            frame_path=frame_path,
            aligned_dir=str(aligned_dir),
            annotated_dir=str(annotated_dir),
        )
        frame_reports.append(report)
        for face in report["faces"]:
            all_aligned_faces.append(face["aligned_face_path"])

    summary: Dict[str, Any] = {
        "module": "Module 2: Preprocessing & ROI Extraction",
        "media_type": parse_result["media_type"],
        "source_path": parse_result["source_path"],
        "output_directory": str(case_dir.resolve()),
        "total_frames_processed": len(frame_reports),
        "total_faces_extracted": len(all_aligned_faces),
        "aligned_face_paths": all_aligned_faces,
        "frames": frame_reports,
        "detector_backend": backend,
    }

    manifest_path = case_dir / "preprocessing_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["manifest_path"] = str(manifest_path.resolve())
    return summary


__all__ = ["FrameParser", "FaceAligner", "run_module2"]