import json
import os
from pathlib import Path
import sys

from PIL import Image
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.module_1.integrity import calculate_hashes, extract_metadata
except ModuleNotFoundError:
    from src.module_1.integrity import calculate_hashes, extract_metadata

from src.module_2 import run_module2

# Page Configuration
st.set_page_config(
    page_title="Multimodal Forensics Toolkit",
    page_icon="🔍",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        margin: 0 auto;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .badge-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #2e3846;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TEMP_DIR = "data/raw"
os.makedirs(TEMP_DIR, exist_ok=True)

if "settings_locked" not in st.session_state:
    st.session_state["settings_locked"] = False
if "applied_settings" not in st.session_state:
    st.session_state["applied_settings"] = {
        "detector_backend": "retinaface",
        "min_confidence": 0.80,
        "output_size": 299,
        "max_frames": 10,
    }

st.title("🔍 Multimodal Explainable Forensics Framework")
st.caption(
    "End-to-End Deepfake and Digital Media Manipulation Forensics Pipeline with High Explainability (XAI)."
)

# =========================================================
# SIDEBAR: CONFIGURATION & INTEGRITY LOCK
# =========================================================
st.sidebar.header("⚙️ 1. Module 2 Settings (AI ROI)")

is_locked = st.session_state["settings_locked"]
current_cfg = st.session_state["applied_settings"]

sel_backend = st.sidebar.selectbox(
    "Face Detection AI Architecture",
    options=["retinaface", "mtcnn"],
    index=0 if current_cfg["detector_backend"] == "retinaface" else 1,
    disabled=is_locked,
    help="RetinaFace (ResNet34) provides superior robustness against false positives in complex backgrounds.",
)

sel_confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=0.99,
    value=float(current_cfg["min_confidence"]),
    step=0.01,
    disabled=is_locked,
)

sel_size = st.sidebar.selectbox(
    "Canonical Alignment Size",
    options=[299, 224],
    index=0 if current_cfg["output_size"] == 299 else 1,
    disabled=is_locked,
    help="299x299 matches the standard input resolution for XceptionNet (Module 4).",
)

sel_max_frames = st.sidebar.slider(
    "Uniform Keyframe Sample Limit (Video)",
    min_value=2,
    max_value=30,
    value=int(current_cfg["max_frames"]),
    step=1,
    disabled=is_locked,
    help="The framework discards blank/monochrome frames and extracts evenly distributed I-frames across the video timeline.",
)

st.sidebar.markdown("---")
# Nút Lock / Unlock điều khiển kích hoạt AI
if not is_locked:
    if st.sidebar.button(
        "🔒 Apply Settings (Lock)", use_container_width=True, type="primary"
    ):
        st.session_state["applied_settings"] = {
            "detector_backend": sel_backend,
            "min_confidence": sel_confidence,
            "output_size": sel_size,
            "max_frames": sel_max_frames,
        }
        st.session_state["settings_locked"] = True
        st.session_state.pop("m2_cache_key", None)
        st.rerun()
else:
    st.sidebar.success("🔒 Configuration is LOCKED for evidence integrity.")
    if st.sidebar.button("🔓 Unlock & Edit Settings", use_container_width=True):
        st.session_state["settings_locked"] = False
        st.session_state["m2_result"] = None  # Xóa kết quả tạm để người dùng chỉnh sửa
        st.session_state.pop("m2_cache_key", None)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📁 2. Evidence Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload target image or video evidence",
    type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
)

# =========================================================
# MAIN INVESTIGATION PIPELINE
# =========================================================
if uploaded_file is not None:
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # =====================================================
    # MODULE 1: EVIDENCE INGESTION & DATA INTEGRITY
    # =====================================================
    st.header("🟢 Module 1: Evidence Ingestion & Chain of Custody")
    col_ev1, col_ev2 = st.columns([1, 1])

    with col_ev1:
        st.subheader("Raw Evidence")
        if uploaded_file.name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            col_l, col_img, col_r = st.columns([0.1, 0.8, 0.1])
            with col_img:
                st.image(Image.open(file_path), caption=uploaded_file.name, width=420)
        else:
            st.video(file_path)

    with col_ev2:
        st.subheader("Cryptographic Hashes (Digital Fingerprint)")
        hashes = calculate_hashes(file_path)
        if hashes:
            st.info(f"**MD5:** `{hashes['MD5']}`\n\n**SHA-256:** `{hashes['SHA-256']}`")

        with st.expander("🔎 Hidden Metadata & EXIF Tags", expanded=False):
            metadata = extract_metadata(file_path)
            if isinstance(metadata, dict):
                clean_meta = {
                    k.split(":")[-1]: v for k, v in metadata.items() if len(str(v)) < 150
                }
                st.json(clean_meta)
            else:
                st.write(metadata)

    # =====================================================
    # MODULE 2: PREPROCESSING & ROI EXTRACTION
    # =====================================================
    st.divider()
    st.header("🟡 Module 2: AI Preprocessing & ROI Extraction")

    # CHẶN AI NẾU NGƯỜI DÙNG CHƯA BẤM "APPLY SETTINGS (LOCK)"
    if not is_locked:
        st.warning(
            "⚠️ **Execution Paused:** Configuration is currently **UNLOCKED**. "
            "Please configure the parameters on the sidebar, then click **'🔒 Apply Settings (Lock)'** to run AI frame extraction and face detection."
        )
    else:
        active_cfg = st.session_state["applied_settings"]
        file_sha = hashes["SHA-256"][:8] if hashes else "nohash"
        cache_key = (
            f"{file_sha}_{active_cfg['detector_backend']}_{active_cfg['min_confidence']:.2f}_"
            f"{active_cfg['output_size']}_{active_cfg['max_frames']}"
        )

        # Chỉ chạy AI khi đã Lock và chưa có trong cache
        if st.session_state.get("m2_cache_key") != cache_key:
            with st.spinner(
                f"Executing frame parsing, blank frame filtering & AI ({active_cfg['detector_backend'].upper()}) Affine Alignment..."
            ):
                try:
                    short_case_id = f"{Path(file_path).stem}_{file_sha}"
                    m2_res = run_module2(
                        input_path=file_path,
                        processed_root="data/processed",
                        case_id=short_case_id,
                        output_size=(active_cfg["output_size"], active_cfg["output_size"]),
                        max_frames=active_cfg["max_frames"],
                        min_confidence=active_cfg["min_confidence"],
                        backend=active_cfg["detector_backend"],
                    )
                    st.session_state["m2_cache_key"] = cache_key
                    st.session_state["m2_result"] = m2_res
                    st.session_state["m2_error"] = None
                except Exception as exc:
                    st.session_state["m2_cache_key"] = cache_key
                    st.session_state["m2_result"] = None
                    st.session_state["m2_error"] = str(exc)

        m2_res = st.session_state.get("m2_result")
        m2_err = st.session_state.get("m2_error")

        if m2_err:
            st.error(f"Module 2 Error: {m2_err}")
        elif m2_res:
            exec_sum = m2_res["execution_summary"]
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Evidence Type", m2_res["forensics_metadata"]["media_type"])
            k2.metric("Detector Model", m2_res["preprocessing_settings"]["ai_detector"])
            k3.metric("Sampled Frames", exec_sum["final_frames_analyzed"])
            k4.metric("Blank/Fade Filtered", exec_sum["discarded_blank_or_extreme_frames"])
            k5.metric("Extracted ROIs", exec_sum["total_faces_extracted"])

            st.caption(f"📁 Persisted artifacts: `{m2_res['artifacts_directory']['root']}`")

            # =================================================
            # INTERACTIVE FRAME & MULTI-MODE INSPECTION
            # =================================================
            total_frames = len(m2_res["frames_detail"])

            if total_frames > 0:
                st.markdown("---")

                # 1. FRAME SELECTOR
                if total_frames > 1:
                    col_sel1, col_sel2 = st.columns([2, 5])
                    with col_sel1:
                        st.markdown("#### 🎞️ Select Target Frame")
                    with col_sel2:
                        selected_frame_idx = st.slider(
                            "Target Frame",
                            min_value=1,
                            max_value=total_frames,
                            value=1,
                            step=1,
                            format="Frame #%d",
                            label_visibility="collapsed",
                        )
                else:
                    selected_frame_idx = 1

                current_frame = m2_res["frames_detail"][selected_frame_idx - 1]

                # 2. SEGMENTED CONTROL MODE SELECTOR
                mode_options = [
                    "👤 1. Aligned Face ROI",
                    "🎯 2. BBox & Landmarks",
                    "🎞️ 3. Raw Keyframe",
                    "📄 4. Frame JSON Record",
                ]

                st.write("")
                if hasattr(st, "segmented_control"):
                    selected_mode = st.segmented_control(
                        "Inspection Mode:",
                        options=mode_options,
                        default=mode_options[0],
                        label_visibility="collapsed",
                    )
                else:
                    selected_mode = st.radio(
                        "Inspection Mode:",
                        options=mode_options,
                        index=0,
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                st.write("")
                st.info(
                    f"**Active Frame:** #{selected_frame_idx} (`{Path(current_frame['frame_path']).name}`) | "
                    f"**Detected Face ROIs:** {current_frame['faces_detected']}"
                )

                # 3. MODE PRESENTATION
                if selected_mode == mode_options[0]:
                    faces = current_frame.get("faces", [])
                    if not faces:
                        st.warning("No face met the detection threshold or geometric sanity check for this frame.")
                    else:
                        for f_info in faces:
                            c_img, c_info = st.columns([1, 2])
                            with c_img:
                                st.image(
                                    f_info["aligned_face_path"],
                                    caption=f"{f_info.get('global_face_id', 'Face')} (Score: {f_info['confidence']:.2f})",
                                    width=260,
                                )
                            with c_info:
                                st.markdown("##### 📌 Geometric & Canonical Metrics")
                                st.markdown(
                                    f"""
                                    - **Global Face ID:** `{f_info.get('global_face_id', 'Face')}`
                                    - **Detection Confidence:** `{f_info['confidence'] * 100:.1f}%`
                                    - **Canonical Dimensions:** `{f_info['aligned_size'][0]} x {f_info['aligned_size'][1]} px`
                                    - **Bounding Box [X1, Y1, X2, Y2]:** `[{f_info['bbox'][0]}, {f_info['bbox'][1]}, {f_info['bbox'][2]}, {f_info['bbox'][3]}]`
                                    - **Landmarks (Eyes):** Left: `({f_info['landmarks']['left_eye'][0]}, {f_info['landmarks']['left_eye'][1]})` | Right: `({f_info['landmarks']['right_eye'][0]}, {f_info['landmarks']['right_eye'][1]})`
                                    - **Landmarks (Nose/Mouth):** Nose: `({f_info['landmarks']['nose'][0]}, {f_info['landmarks']['nose'][1]})` | Mouth Left: `({f_info['landmarks']['mouth_left'][0]}, {f_info['landmarks']['mouth_left'][1]})`
                                    """
                                )
                            st.markdown("---")

                elif selected_mode == mode_options[1]:
                    ann_path = current_frame.get("annotated_frame_path")
                    if ann_path and os.path.exists(ann_path):
                        col_l, col_center, col_r = st.columns([1, 2, 1])
                        with col_center:
                            st.image(
                                ann_path,
                                caption=f"AI Localization Overlay: {Path(ann_path).name}",
                                width=480,
                            )
                    else:
                        st.info("No annotated frame visualization available.")

                elif selected_mode == mode_options[2]:
                    col_l, col_center, col_r = st.columns([1, 2, 1])
                    with col_center:
                        st.image(
                            current_frame["frame_path"],
                            caption=f"Raw Keyframe: {Path(current_frame['frame_path']).name}",
                            width=480,
                        )

                elif selected_mode == mode_options[3]:
                    st.markdown("##### 📄 Frame-Level Forensic Record (JSON)")
                    st.json(current_frame)

            # =================================================
            # MANIFEST SUMMARY & AUDIT TRAIL
            # =================================================
            st.divider()
            st.subheader("📋 Preprocessing Manifest & Audit Summary")
            c_sum1, c_sum2 = st.columns([1, 1])
            with c_sum1:
                st.table({
                    "Forensic Property": [
                        "Case Identifier",
                        "Audit Timestamp",
                        "Sampling Mode",
                        "Canonical Resolution",
                        "Average Faces / Frame",
                    ],
                    "Observed Value": [
                        m2_res["forensics_metadata"]["case_id"],
                        m2_res["forensics_metadata"]["timestamp"],
                        m2_res["preprocessing_settings"]["sampling_mode"],
                        m2_res["preprocessing_settings"]["canonical_resolution"],
                        f"{exec_sum['average_faces_per_frame']} faces/frame",
                    ],
                })
            with c_sum2:
                manifest_json_str = json.dumps(m2_res, ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Download preprocessing_manifest.json",
                    data=manifest_json_str,
                    file_name=f"{m2_res['forensics_metadata']['case_id']}_manifest.json",
                    mime="application/json",
                    type="primary",
                )
                with st.expander("View Raw Manifest JSON", expanded=False):
                    st.code(manifest_json_str, language="json")
else:
    st.info(
        "👈 Please configure settings in the left sidebar, click 'Apply Settings (Lock)', and upload evidence to begin analysis."
    )