from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple

from .face_aligner import FaceAligner
from .frame_parser import FrameParser


def run_module2(
    input_path: str,
    processed_root: str = "data/processed",
    case_id: Optional[str] = None,
    output_size: Tuple[int, int] = (299, 299),
    max_frames: Optional[int] = 10,
    min_confidence: float = 0.80,
    backend: str = "retinaface",
) -> Dict[str, Any]:
    """
    Pipeline Module 2: Preprocessing & ROI Extraction
    - Tự động tạo thư mục case và lưu ngay lập tức xuống máy.
    - Xuất file preprocessing_manifest.json có cấu trúc dễ đọc, phân tầng rõ ràng.
    """
    src = Path(input_path)
    folder_name = case_id or src.stem
    case_dir = Path(processed_root) / folder_name
    frames_dir = case_dir / "frames"
    aligned_dir = case_dir / "aligned_faces"
    annotated_dir = case_dir / "annotated"

    # Làm sạch thư mục cũ của case này nếu có
    for sub_dir in (frames_dir, aligned_dir, annotated_dir):
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

    # 1. Frame Parsing (Lọc I-frame, lọc đen/trắng, rải đều)
    parser = FrameParser(max_frames=max_frames)
    parse_result = parser.parse(str(src), str(frames_dir))

    # 2. Face Alignment
    aligner = FaceAligner(
        backend=backend,
        output_size=output_size,
        min_confidence=min_confidence,
    )

    frame_reports = []
    all_aligned_faces = []
    face_global_counter = 0

    for frame_idx, frame_path in enumerate(parse_result["frames"]):
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

        frame_reports.append(report)

    # 3. Tạo cấu trúc Manifest thân thiện người dùng
    manifest_data: Dict[str, Any] = {
        "forensics_metadata": {
            "module": "Module 2: Preprocessing & ROI Extraction",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": folder_name,
            "source_file": src.name,
            "source_path": str(src.resolve()),
            "media_type": parse_result["media_type"].upper(),
        },
        "preprocessing_settings": {
            "ai_detector": backend.upper(),
            "confidence_threshold": min_confidence,
            "canonical_resolution": f"{output_size[0]}x{output_size[1]}",
            "sampling_mode": parse_result.get("sampling_mode", "Standard"),
            "target_frames_limit": max_frames,
        },
        "execution_summary": {
            "total_frames_inspected": parse_result.get("total_raw_keyframes", len(parse_result["frames"])),
            "discarded_blank_or_extreme_frames": parse_result.get("discarded_blank_frames", 0),
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

    # Ghi file manifest.json ngay lập tức xuống đĩa cứng
    manifest_path = case_dir / "preprocessing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    manifest_data["manifest_path"] = str(manifest_path.resolve())
    return manifest_data


__all__ = ["FrameParser", "FaceAligner", "run_module2"]