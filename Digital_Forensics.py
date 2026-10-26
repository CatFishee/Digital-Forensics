
import sys
import torch
import cv2
import numpy as np
try:
    import exiftool
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

def check_environment():
    print("="*50)
    print(" KIỂM TRA MÔI TRƯỜNG ĐỒ ÁN DIGITAL FORENSICS ")
    print("="*50)
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
        print(f"\n[+] PyExifTool wrapper: Đã cài đặt")
    else:
        print(f"\n[!] PyExifTool wrapper: CHƯA cài đặt")
        
    print("="*50)

if __name__ == "__main__":
    check_environment()