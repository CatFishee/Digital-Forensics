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
    Tiền xử lý: Phát hiện frame bị đen, trắng hoàn toàn hoặc đơn sắc (monochrome/blank).
    - black_thresh: độ sáng trung bình <= 10 -> coi là frame đen (cắt cảnh, fade to black).
    - white_thresh: độ sáng trung bình >= 245 -> coi là frame trắng (chớp sáng, fade to white).
    - std_thresh: độ lệch chuẩn <= 3.5 -> frame trơn một màu, không có chi tiết cảnh vật.
    """
    if img_bgr is None or img_bgr.size == 0:
        return True, "Frame rỗng (0 byte hoặc không đọc được)"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val <= black_thresh:
        return True, f"Frame đen hoàn toàn (Mean: {mean_val:.1f} <= {black_thresh})"
    if mean_val >= white_thresh:
        return True, f"Frame trắng hoàn toàn (Mean: {mean_val:.1f} >= {white_thresh})"
    if std_val <= std_thresh:
        return True, f"Frame đơn sắc không có chi tiết (Std: {std_val:.1f} <= {std_thresh})"

    return False, "Hợp lệ"


class FrameParser:
    """
    Module 2.1: Frame Parsing chuyên sâu
    - Lọc I-frames từ video.
    - Loại bỏ frame đen / trắng hoàn toàn trước khi đưa vào AI.
    - Lấy mẫu rải đều (Uniform Sampling) trên toàn bộ chiều dài video.
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
            raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {src}")

        media_type = self.detect_media_type(src)
        out_dir = Path(output_frames_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if media_type == "image":
            # Đọc kiểm tra ảnh tĩnh xem có bị đen/trắng không
            img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            is_blank, reason = is_blank_or_extreme_frame(img)
            if is_blank:
                raise ValueError(f"Ảnh đầu vào không hợp lệ: {reason}")

            return {
                "media_type": "image",
                "source_path": str(src.resolve()),
                "frames": [str(src.resolve())],
                "total_frames_extracted": 1,
                "discarded_blank_frames": 0,
                "sampling_mode": "Single Image",
                "note": "Ảnh tĩnh hợp lệ được chuyển thẳng vào AI.",
            }

        if media_type == "video":
            if shutil.which("ffmpeg") is None:
                raise EnvironmentError("Không tìm thấy 'ffmpeg.exe' trong PATH hệ thống.")

            # Thư mục tạm chứa toàn bộ keyframe trước khi lọc & rải đều
            raw_keyframes_dir = out_dir / "_raw_keyframes"
            if raw_keyframes_dir.exists():
                shutil.rmtree(raw_keyframes_dir)
            raw_keyframes_dir.mkdir(parents=True, exist_ok=True)

            raw_pattern = raw_keyframes_dir / "raw_%05d.png"

            # 1. Trích xuất toàn bộ I-frames có trong video
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
                raise RuntimeError(f"Lỗi FFmpeg khi trích xuất I-frames: {err_msg}") from exc

            raw_files = sorted(raw_keyframes_dir.glob("raw_*.png"))
            if not raw_files:
                raise RuntimeError("Video không có I-frame nào trích xuất được.")

            # 2. Tiền xử lý lọc bỏ frame đen/trắng
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
                # Nếu video toàn cảnh fade, dự phòng giữ lại frame gốc có độ sáng tốt nhất
                valid_candidates = raw_files

            # 3. Thuật toán Rải đều (Uniform Sampling) khắp chiều dài video
            total_valid = len(valid_candidates)
            target_count = min(self.max_frames, total_valid)

            if total_valid <= target_count:
                selected_paths = valid_candidates
            else:
                # Trải đều từ frame đầu tiên đến frame cuối cùng
                indices = np.round(np.linspace(0, total_valid - 1, target_count)).astype(int)
                selected_paths = [valid_candidates[i] for i in indices]

            # 4. Lưu chính thức vào thư mục frames/
            for old_file in out_dir.glob("iframe_*.png"):
                old_file.unlink()

            final_frames: List[str] = []
            for idx, p in enumerate(selected_paths):
                target_file = out_dir / f"iframe_{idx + 1:05d}.png"
                shutil.copy2(p, target_file)
                final_frames.append(str(target_file.resolve()))

            # Dọn thư mục tạm
            shutil.rmtree(raw_keyframes_dir, ignore_errors=True)

            return {
                "media_type": "video",
                "source_path": str(src.resolve()),
                "frames": final_frames,
                "total_frames_extracted": len(final_frames),
                "total_raw_keyframes": len(raw_files),
                "discarded_blank_frames": discarded_count,
                "sampling_mode": f"Uniform Sampling ({len(final_frames)}/{len(raw_files)} I-frames)",
                "note": "Đã loại bỏ frame đen/trắng và phân bổ đều khung hình theo dòng thời gian video.",
            }

        raise ValueError(f"Định dạng tệp chưa được hỗ trợ: {src.suffix}")