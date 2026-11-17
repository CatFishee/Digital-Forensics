#!/usr/bin/env bash
set -e

# Luôn chạy tại thư mục chứa file start.sh (an toàn kể cả khi gọi từ chỗ khác)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
HASH_FILE="$VENV_DIR/.requirements.sha256"

echo "=========================================="
echo "  DEEPFAKE FORENSIC TOOLKIT - LAUNCHER"
echo "=========================================="

# 1. Tạo venv nếu chưa có, sau đó kích hoạt venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Chưa thấy venv, đang tạo mới..."
    python3 -m venv "$VENV_DIR"
fi

echo "[1/3] Đang kích hoạt môi trường ảo (venv)..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2. Cài đặt / cập nhật thư viện (chỉ chạy pip khi requirements.txt thay đổi hoặc truyền --install)
CURRENT_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
SAVED_HASH=""
if [ -f "$HASH_FILE" ]; then
    SAVED_HASH="$(cat "$HASH_FILE")"
fi

if [ "$1" == "--install" ] || [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
    echo "[2/3] Phát hiện thay đổi ở requirements.txt -> Đang chạy pip install..."
    python -m pip install --upgrade pip
    pip install -r "$REQ_FILE"
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "[2/3] requirements.txt không đổi -> Bỏ qua pip install (dùng './start.sh --install' để ép cài lại)."
fi

# 3. Khởi chạy Streamlit (dùng poll để tránh lỗi Inotify queue overflowed trên Linux)
echo "[3/3] Đang khởi động Streamlit..."
exec streamlit run "$SCRIPT_DIR/dashboard.py" --server.fileWatcherType poll
