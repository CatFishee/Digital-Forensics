import hashlib
import os
try:
    import exiftool
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

def calculate_hashes(file_path):
    """Tính mã băm MD5 và SHA-256 để đảm bảo tính toàn vẹn (Chain of Custody)"""
    if not os.path.exists(file_path):
        return None
    
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    
    with open(file_path, "rb") as f:
        # Đọc theo từng block để tránh tràn RAM với file video lớn
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            md5_hash.update(byte_block)
            
    return {
        "MD5": md5_hash.hexdigest(),
        "SHA-256": sha256_hash.hexdigest()
    }

def extract_metadata(file_path):
    """Trích xuất EXIF và Metadata ẩn bằng ExifTool"""
    if not HAS_EXIF or not os.path.exists(file_path):
        return {"Error": "ExifTool chưa được cài đặt hoặc file không tồn tại."}
    
    try:
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)
            if metadata:
                return metadata[0] # Trả về dictionary chứa metadata của file đầu tiên
            return {"Info": "Không tìm thấy metadata."}
    except Exception as e:
        return {"Error": f"Lỗi khi đọc metadata: {str(e)}"}