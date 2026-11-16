import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import ffmpeg

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


class FrameParser:
    """
    Module 2.1: Frame Parsing
    - Nếu đầu vào là Ảnh: giữ nguyên file gốc (tránh nén lại gây nhiễu cho Module 3).
    - Nếu đầu vào là Video: trích xuất các I-frames (Keyframes) ra định dạng PNG không nén.
    """

    def __init__(self, max_frames: Optional[int] = 30):
        self.max_frames = max_frames

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
            return {
                "media_type": "image",
                "source_path": str(src.resolve()),
                "frames": [str(src.resolve())],
                "total_frames_extracted": 1,
                "note": "Ảnh tĩnh đầu vào được giữ nguyên trực tiếp (không giải mã/nén lại).",
            }

        if media_type == "video":
            if shutil.which("ffmpeg") is None:
                raise EnvironmentError(
                    "Không tìm thấy 'ffmpeg.exe' trong PATH. "
                    "Hãy cài bằng lệnh: winget install Gyan.FFmpeg rồi mở lại terminal."
                )

            output_pattern = out_dir / "iframe_%05d.png"
            # Xóa các I-frame cũ nếu chạy lại cùng thư mục
            for old_file in out_dir.glob("iframe_*.png"):
                old_file.unlink()

            output_kwargs: Dict[str, Any] = {"fps_mode": "vfr"}
            if self.max_frames is not None and self.max_frames > 0:
                output_kwargs["frames:v"] = int(self.max_frames)

            try:
                (
                    ffmpeg.input(str(src))
                    .filter("select", "eq(pict_type,I)")
                    .output(str(output_pattern), **output_kwargs)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as exc:
                err_msg = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
                raise RuntimeError(f"Lỗi FFmpeg khi trích xuất I-frames: {err_msg}") from exc

            extracted_frames: List[str] = [
                str(p.resolve()) for p in sorted(out_dir.glob("iframe_*.png"))
            ]

            return {
                "media_type": "video",
                "source_path": str(src.resolve()),
                "frames": extracted_frames,
                "total_frames_extracted": len(extracted_frames),
                "note": "Đã trích xuất các I-frames (Keyframes) ra PNG không nén.",
            }

        raise ValueError(f"Định dạng tệp chưa được hỗ trợ: {src.suffix}")