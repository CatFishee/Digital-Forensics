import hashlib
import os

try:
    import exiftool
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False


def calculate_hashes(file_path):
    """Compute MD5 and SHA-256 hashes to establish Chain of Custody integrity."""
    if not os.path.exists(file_path):
        return None

    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        # Read in 4KB blocks to avoid memory spikes on large video files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            md5_hash.update(byte_block)

    return {
        "MD5": md5_hash.hexdigest(),
        "SHA-256": sha256_hash.hexdigest(),
    }


def extract_metadata(file_path):
    """Extract EXIF and hidden metadata using ExifTool."""
    if not HAS_EXIF or not os.path.exists(file_path):
        return {"Error": "ExifTool is not installed or the file does not exist."}

    try:
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)
            if metadata:
                return metadata[0]
            return {"Info": "No metadata found."}
    except Exception as e:
        return {"Error": f"Failed to read metadata: {str(e)}"}
