import streamlit as st
import os
from src.ingestion.integrity import calculate_hashes, extract_metadata
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="Digital Forensics Toolkit", page_icon="🔍", layout="wide")

# Đảm bảo thư mục lưu file tạm tồn tại
TEMP_DIR = "data/raw"
os.makedirs(TEMP_DIR, exist_ok=True)

st.title("🔍 Digital Forensics Toolkit")
st.markdown("Hệ thống điều tra ảnh/video giả mạo & Deepfake")

# Sidebar tải file
st.sidebar.header("Tải chứng cứ (Evidence)")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh hoặc video", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    # 1. Lưu file upload xuống ổ cứng (để ExifTool có đường dẫn đọc)
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"Đã tải lên: {uploaded_file.name}")
    
    # Chia cột hiển thị UI
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Ảnh bằng chứng (Evidence Image)")
        if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file_path)
            st.image(image, use_column_width=True)
            
    with col2:
        st.subheader("2. Tính toàn vẹn số (Digital Integrity)")
        hashes = calculate_hashes(file_path)
        if hashes:
            st.info(f"**MD5:** {hashes['MD5']}\n\n**SHA-256:** {hashes['SHA-256']}")
        
        st.subheader("3. Dữ liệu siêu ẩn (Metadata / EXIF)")
        with st.expander("Bấm để xem chi tiết Metadata", expanded=True):
            metadata = extract_metadata(file_path)
            # Lọc bớt các tag quá dài của exiftool để UI gọn gàng
            clean_meta = {k.split(':')[-1]: v for k, v in metadata.items() if len(str(v)) < 100}
            st.json(clean_meta)
else:
    st.info("👈 Vui lòng tải một tệp ảnh (jpg, png) lên từ menu bên trái để bắt đầu phân tích.")