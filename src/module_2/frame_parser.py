import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def is_blank_or_extreme_frame(
    img_bgr: np.ndarray,
    black_thresh: float = 10.0,
    white_thresh: float = 245.0,
    std_thresh: float = 3.5,
) -> Tuple[bool, str]:
    """
    Pre-filter: Detect completely black, completely white, or monochrome/blank frames.
    """
    if img_bgr is None or img_bgr.size == 0:
        return True, "Empty frame (0 bytes or unreadable)"

    gray = (
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if len(img_bgr.shape) == 3
        else img_bgr
    )
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val <= black_thresh:
        return True, f"Completely black frame (Mean: {mean_val:.1f} <= {black_thresh})"
    if mean_val >= white_thresh:
        return True, f"Completely white frame (Mean: {mean_val:.1f} >= {white_thresh})"
    if std_val <= std_thresh:
        return (
            True,
            f"Monochrome frame with no detail (Std: {std_val:.1f} <= {std_thresh})",
        )

    return False, "Valid"


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


class FrameParser:
    """
    Module 2.1: Frame & Consecutive Burst Parsing
    - Samples evenly spaced non-blank frames across the full video timeline.
    - Guarantees the requested frame count whenever enough non-blank frames exist
      (replaces discarded blank frames automatically).
    - Extracts a short consecutive frame burst (default: 5 frames) around each
      sampled frame for Temporal Face Flicker analysis in Module 3.
    """

    def __init__(self, max_frames: Optional[int] = 10, burst_length: int = 5):
        self.max_frames = max(1, int(max_frames or 10))
        self.burst_length = max(1, int(burst_length))

    @staticmethod
    def detect_media_type(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"
        return "unknown"

    def _scan_video_candidates(
        self, video_path: Path
    ) -> Tuple[List[int], int, int, float]:
        """
        Return (valid_non_blank_indices, discarded_blank_count, actual_total_frames, fps).
        Uses a fast grab-pass probe first, then falls back to a denser scan if blank
        frames caused fewer valid candidates than self.max_frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            est_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()

        def _probe(target_indices: Optional[set[int]], stride: int = 1):
            c = cv2.VideoCapture(str(video_path))
            valid_idx: List[int] = []
            blank_cnt = 0
            idx = 0
            try:
                while True:
                    grabbed = c.grab()
                    if not grabbed:
                        break
                    should_decode = (
                        target_indices is None and (idx % stride == 0)
                    ) or (target_indices is not None and idx in target_indices)
                    if should_decode:
                        ok, frame = c.retrieve()
                        if not ok or frame is None:
                            blank_cnt += 1
                        else:
                            is_blank, _ = is_blank_or_extreme_frame(frame)
                            if is_blank:
                                blank_cnt += 1
                            else:
                                valid_idx.append(idx)
                    idx += 1
            finally:
                c.release()
            return valid_idx, blank_cnt, idx

        if est_total > 0:
            probe_count = min(est_total, max(self.max_frames * 4, 48))
            probe_indices = set(
                np.unique(
                    np.round(np.linspace(0, est_total - 1, probe_count)).astype(int)
                ).tolist()
            )
            valid_idx, blank_cnt, actual_total = _probe(probe_indices)

            if len(valid_idx) < self.max_frames and len(probe_indices) < actual_total:
                dense_stride = 1 if actual_total <= 1800 else max(1, actual_total // 1800)
                valid_idx, blank_cnt, actual_total = _probe(None, stride=dense_stride)
                if len(valid_idx) < self.max_frames and dense_stride > 1:
                    valid_idx, blank_cnt, actual_total = _probe(None, stride=1)
        else:
            valid_idx, blank_cnt, actual_total = _probe(None, stride=1)

        return valid_idx, blank_cnt, actual_total, fps

    def parse(self, input_path: str, output_frames_dir: str) -> Dict[str, Any]:
        src = Path(input_path)
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")

        media_type = self.detect_media_type(src)
        out_dir = Path(output_frames_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if media_type == "image":
            img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            is_blank, reason = is_blank_or_extreme_frame(img)
            if is_blank:
                raise ValueError(f"Invalid input image: {reason}")

            resolved = str(src.resolve())
            return {
                "media_type": "image",
                "source_path": resolved,
                "frames": [resolved],
                "bursts": [[resolved]],
                "burst_length": 1,
                "fps": None,
                "total_frames_extracted": 1,
                "total_raw_keyframes": 1,
                "discarded_blank_frames": 0,
                "sampling_mode": "Single Image",
                "note": "Valid static image passed directly to downstream modules.",
            }

        if media_type == "video":
            valid_indices, discarded_count, actual_total, fps = (
                self._scan_video_candidates(src)
            )
            if actual_total == 0:
                raise RuntimeError("No frames could be decoded from the video.")

            if not valid_indices:
                valid_indices = (
                    np.round(
                        np.linspace(
                            0, actual_total - 1, min(self.max_frames, actual_total)
                        )
                    )
                    .astype(int)
                    .tolist()
                )

            total_valid = len(valid_indices)
            target_count = min(self.max_frames, total_valid)
            pick_pos = np.round(np.linspace(0, total_valid - 1, target_count)).astype(
                int
            )
            anchor_indices = [valid_indices[p] for p in pick_pos]

            burst_index_lists: List[List[int]] = []
            needed_indices: set[int] = set()
            for a_idx in anchor_indices:
                start = min(a_idx, max(0, actual_total - self.burst_length))
                end = min(actual_total, start + self.burst_length)
                b_list = list(range(start, end))
                if a_idx not in b_list:
                    b_list = [a_idx] + [x for x in b_list if x != a_idx][
                        : self.burst_length - 1
                    ]
                b_list.sort()
                burst_index_lists.append(b_list)
                needed_indices.update(b_list)

            decoded_frames: Dict[int, np.ndarray] = {}
            cap = cv2.VideoCapture(str(src))
            cur_idx = 0
            max_needed = max(needed_indices) if needed_indices else -1
            try:
                while cur_idx <= max_needed:
                    grabbed = cap.grab()
                    if not grabbed:
                        break
                    if cur_idx in needed_indices:
                        ok, frame = cap.retrieve()
                        if ok and frame is not None:
                            decoded_frames[cur_idx] = frame
                    cur_idx += 1
            finally:
                cap.release()

            for old_file in out_dir.glob("iframe_*.png"):
                old_file.unlink()
            bursts_dir = out_dir / "bursts"
            if bursts_dir.exists():
                shutil.rmtree(bursts_dir)
            bursts_dir.mkdir(parents=True, exist_ok=True)

            final_frames: List[str] = []
            final_bursts: List[List[str]] = []

            for idx, (a_idx, b_list) in enumerate(
                zip(anchor_indices, burst_index_lists)
            ):
                anchor_img = decoded_frames.get(a_idx)
                if anchor_img is None:
                    continue

                anchor_file = out_dir / f"iframe_{idx + 1:05d}.png"
                _imwrite_unicode(anchor_file, anchor_img)
                anchor_resolved = str(anchor_file.resolve())
                final_frames.append(anchor_resolved)

                burst_paths: List[str] = []
                for b_pos, frame_num in enumerate(b_list):
                    b_img = decoded_frames.get(frame_num)
                    if b_img is None:
                        continue
                    if frame_num != a_idx:
                        is_blank, _ = is_blank_or_extreme_frame(b_img)
                        if is_blank:
                            continue
                    burst_file = bursts_dir / f"frame_{idx + 1:05d}_b{b_pos:02d}.png"
                    if frame_num == a_idx:
                        shutil.copy2(anchor_file, burst_file)
                    else:
                        _imwrite_unicode(burst_file, b_img)
                    burst_paths.append(str(burst_file.resolve()))

                if not burst_paths:
                    burst_paths = [anchor_resolved]
                final_bursts.append(burst_paths)

            return {
                "media_type": "video",
                "source_path": str(src.resolve()),
                "frames": final_frames,
                "bursts": final_bursts,
                "burst_length": self.burst_length,
                "fps": round(fps, 3) if fps > 0 else None,
                "total_frames_extracted": len(final_frames),
                "total_raw_keyframes": actual_total,
                "discarded_blank_frames": discarded_count,
                "sampling_mode": (
                    f"Evenly Spaced + {self.burst_length}-Frame Bursts "
                    f"({len(final_frames)}/{self.max_frames} frames)"
                ),
                "note": (
                    "Discarded blank frames, automatically topped up to match the requested frame count, "
                    "and extracted short consecutive bursts for temporal flicker analysis."
                ),
            }

        raise ValueError(f"Unsupported file format: {src.suffix}")
