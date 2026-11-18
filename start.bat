@echo off
chcp 65001 >nul
setlocal

:: Luôn chạy tại thư mục chứa file start.bat (an toàn kể cả khi gọi từ chỗ khác)
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

set "VENV_DIR=%SCRIPT_DIR%\venv"
set "REQ_FILE=%SCRIPT_DIR%\requirements.txt"
set "HASH_FILE=%VENV_DIR%\.requirements.sha256"

echo ==========================================
echo   DEEPFAKE FORENSIC TOOLKIT - LAUNCHER
echo ==========================================

:: 1. Tạo venv nếu chưa có (ưu tiên Python 3.12 / 3.13 để tránh lỗi build pyjpegio), sau đó kích hoạt
if exist "%VENV_DIR%\Scripts\python.exe" goto :activate_venv

echo [1/3] Chưa thấy venv, đang tạo mới...
set "PY_CMD="
py -3.12 --version >nul 2>&1 && set "PY_CMD=py -3.12"
if not defined PY_CMD py -3.13 --version >nul 2>&1 && set "PY_CMD=py -3.13"
if not defined PY_CMD py -3.11 --version >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD py -3.10 --version >nul 2>&1 && set "PY_CMD=py -3.10"
if not defined PY_CMD where python >nul 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
    echo [LỖI] Không tìm thấy Python! Vui lòng cài đặt Python 3.10 - 3.13.
    goto :fail
)

echo       Đang dùng: %PY_CMD%
%PY_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [LỖI] Không thể tạo venv.
    goto :fail
)

:activate_venv
echo [1/3] Đang kích hoạt môi trường ảo (venv)...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [LỖI] Không thể kích hoạt venv.
    goto :fail
)

:: 2. Cài đặt / cập nhật thư viện (chỉ chạy pip khi requirements.txt thay đổi hoặc truyền --install)
if not exist "%REQ_FILE%" (
    echo [LỖI] Không tìm thấy file requirements.txt!
    goto :fail
)

set "CURRENT_HASH="
for /f "skip=1 tokens=1" %%H in ('certutil -hashfile "%REQ_FILE%" SHA256') do (
    if not defined CURRENT_HASH set "CURRENT_HASH=%%H"
)

set "SAVED_HASH="
if exist "%HASH_FILE%" (
    set /p SAVED_HASH=<"%HASH_FILE%"
)

set "NEED_INSTALL=0"
if /i "%~1"=="--install" set "NEED_INSTALL=1"
if /i not "%CURRENT_HASH%"=="%SAVED_HASH%" set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
    echo [2/3] Phát hiện thay đổi ở requirements.txt -^> Đang chạy pip install...
    python -m pip install --upgrade pip
    if errorlevel 1 goto :fail
    python -m pip install -r "%REQ_FILE%"
    if errorlevel 1 goto :fail
    >"%HASH_FILE%" echo %CURRENT_HASH%
) else (
    echo [2/3] requirements.txt không đổi -^> Bỏ qua pip install ^(dùng 'start.bat --install' để ép cài lại^).
)

:: 3. Khởi chạy Streamlit
echo [3/3] Đang khởi động Streamlit...
streamlit run "%SCRIPT_DIR%\dashboard.py" --server.fileWatcherType poll
if errorlevel 1 goto :fail

exit /b 0

:fail
echo.
echo [THẤT BẠI] Quá trình khởi chạy gặp lỗi. Nhấn phím bất kỳ để đóng...
pause >nul
exit /b 1