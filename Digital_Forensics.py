import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

try:
    import exiftool
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

from src.ingestion.integrity import calculate_hashes, extract_metadata
from src.Module2 import run_module2


def check_environment():
    print("=" * 50)
    print(" KIỂM TRA MÔI TRƯỜNG ĐỒ ÁN DIGITAL FORENSICS ")
    print("=" * 50)
    print(f"[+] Python version: {sys.version.split(' ')[0]}")

    # 1. Check Deep Learning
    print(f"\n[+] PyTorch version: {torch.__version__}")
    print(f"[-] GPU (CUDA) Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[-] GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Check Computer Vision & Math
    print(f"\n[+] OpenCV version: {cv2.__version__}")
    print(f"[+] NumPy version: {np.__version__}")

    # 3. Check Thư viện Python cho ExifTool
    if HAS_EXIF:
        print("\n[+] PyExifTool wrapper: Đã cài đặt")
    else:
        print("\n[!] PyExifTool wrapper: CHƯA cài đặt")

    print("=" * 50)


def run_pipeline(input_path: str, max_frames: int = 30, size: int = 299):
    file_path = Path(input_path)
    if not file_path.exists():
        print(f"[!] Không tìm thấy tệp chứng cứ: {file_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f" BẮT ĐẦU ĐIỀU TRA CHỨNG CỨ: {file_path.name}")
    print("=" * 60)

    # --- MODULE 1: EVIDENCE INGESTION & INTEGRITY ---
    print("\n[MODULE 1] Khởi tạo & Bảo toàn tính toàn vẹn...")
    hashes = calculate_hashes(str(file_path))
    metadata = extract_metadata(str(file_path))
    if hashes:
        print(f"  - MD5    : {hashes['MD5']}")
        print(f"  - SHA-256: {hashes['SHA-256']}")
    if isinstance(metadata, dict) and "Error" in metadata:
        print(f"  - Metadata: {metadata['Error']}")
    else:
        print(f"  - Metadata: Đã trích xuất {len(metadata)} thẻ.")

    # --- MODULE 2: PREPROCESSING & ROI EXTRACTION ---
    print("\n[MODULE 2] Tiền xử lý & Trích xuất vùng quan tâm (ROI)...")
    short_case_id = f"{file_path.stem}_{hashes['SHA-256'][:8]}" if hashes else file_path.stem
    m2_result = run_module2(
        input_path=str(file_path),
        processed_root="data/processed",
        case_id=short_case_id,
        output_size=(size, size),
        max_frames=max_frames,
    )

    print(f"  - Loại dữ liệu    : {m2_result['media_type'].upper()}")
    print(f"  - Số frame xử lý  : {m2_result['total_frames_processed']}")
    print(f"  - Số khuôn mặt cắt: {m2_result['total_faces_extracted']}")
    for face_img in m2_result["aligned_face_paths"]:
        print(f"    -> Aligned Face : {face_img}")
    print(f"  - Manifest JSON   : {m2_result['manifest_path']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Explainable Forensics Framework")
    parser.add_argument("--input", "-i", type=str, help="Đường dẫn ảnh hoặc video cần phân tích")
    parser.add_argument("--max-frames", type=int, default=30, help="Số I-frame tối đa trích xuất từ video")
    parser.add_argument("--size", type=int, default=299, help="Kích thước khuôn mặt sau Affine Alignment")
    args = parser.parse_args()

    if args.input:
        run_pipeline(args.input, max_frames=args.max_frames, size=args.size)
    else:
        check_environment()