import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import ffmpeg
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
    - black_thresh: mean luminance <= 10 -> black frame (scene cut, fade to black).
    - white_thresh: mean luminance >= 245 -> white frame (flash, fade to white).
    - std_thresh: standard deviation <= 3.5 -> flat monochrome frame without visual detail.
    """
    if img_bgr is None or img_bgr.size == 0:
        return True, "Empty frame (0 bytes or unreadable)"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val <= black_thresh:
        return True, f"Completely black frame (Mean: {mean_val:.1f} <= {black_thresh})"
    if mean_val >= white_thresh:
        return True, f"Completely white frame (Mean: {mean_val:.1f} >= {white_thresh})"
    if std_val <= std_thresh:
        return True, f"Monochrome frame with no detail (Std: {std_val:.1f} <= {std_thresh})"

    return False, "Valid"


class FrameParser:
    """
    Module 2.1: Frame Parsing
    - Extracts I-frames from video evidence.
    - Filters out black/white/monochrome frames before AI analysis.
    - Applies uniform sampling across the entire video timeline.
    """

    def __init__(self, max_frames: Optional[int] = 10):
        self.max_frames = max_frames or 10

    @staticmethod
    def detect_media_type(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"
        return "unknown"

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

            return {
                "media_type": "image",
                "source_path": str(src.resolve()),
                "frames": [str(src.resolve())],
                "total_frames_extracted": 1,
                "discarded_blank_frames": 0,
                "sampling_mode": "Single Image",
                "note": "Valid static image passed directly to downstream modules.",
            }

        if media_type == "video":
            if shutil.which("ffmpeg") is None:
                raise EnvironmentError("'ffmpeg' executable was not found in system PATH.")

            raw_keyframes_dir = out_dir / "_raw_keyframes"
            if raw_keyframes_dir.exists():
                shutil.rmtree(raw_keyframes_dir)
            raw_keyframes_dir.mkdir(parents=True, exist_ok=True)

            raw_pattern = raw_keyframes_dir / "raw_%05d.png"

            # 1. Extract all I-frames from the video
            try:
                (
                    ffmpeg.input(str(src))
                    .filter("select", "eq(pict_type,I)")
                    .output(str(raw_pattern), fps_mode="vfr")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as exc:
                err_msg = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
                raise RuntimeError(f"FFmpeg error while extracting I-frames: {err_msg}") from exc

            raw_files = sorted(raw_keyframes_dir.glob("raw_*.png"))
            if not raw_files:
                raise RuntimeError("No I-frames could be extracted from the video.")

            # 2. Filter out blank/extreme frames
            valid_candidates: List[Path] = []
            discarded_count = 0
            for kf in raw_files:
                img = cv2.imdecode(np.fromfile(str(kf), dtype=np.uint8), cv2.IMREAD_COLOR)
                is_blank, _ = is_blank_or_extreme_frame(img)
                if is_blank:
                    discarded_count += 1
                else:
                    valid_candidates.append(kf)

            if not valid_candidates:
                # Fallback if all frames were fade/blank
                valid_candidates = raw_files

            # 3. Uniform sampling across the full video duration
            total_valid = len(valid_candidates)
            target_count = min(self.max_frames, total_valid)

            if total_valid <= target_count:
                selected_paths = valid_candidates
            else:
                indices = np.round(np.linspace(0, total_valid - 1, target_count)).astype(int)
                selected_paths = [valid_candidates[i] for i in indices]

            # 4. Save final frames into frames/
            for old_file in out_dir.glob("iframe_*.png"):
                old_file.unlink()

            final_frames: List[str] = []
            for idx, p in enumerate(selected_paths):
                target_file = out_dir / f"iframe_{idx + 1:05d}.png"
                shutil.copy2(p, target_file)
                final_frames.append(str(target_file.resolve()))

            shutil.rmtree(raw_keyframes_dir, ignore_errors=True)

            return {
                "media_type": "video",
                "source_path": str(src.resolve()),
                "frames": final_frames,
                "total_frames_extracted": len(final_frames),
                "total_raw_keyframes": len(raw_files),
                "discarded_blank_frames": discarded_count,
                "sampling_mode": f"Uniform Sampling ({len(final_frames)}/{len(raw_files)} I-frames)",
                "note": "Removed blank/extreme frames and distributed keyframes uniformly across the video timeline.",
            }

        raise ValueError(f"Unsupported file format: {src.suffix}")
