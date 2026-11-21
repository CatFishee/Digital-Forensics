from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module_1 import calculate_hashes, extract_metadata
from src.module_2 import run_module2
from src.module_3 import (
    ELAAnalyzer,
    FFTAnalyzer,
    Module3ForensicTranslator,
    NoiseAnalyzer,
)

TEMP_DIR = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def _imread_unicode(path: Path) -> np.ndarray | None:
    """Safely read an image from disk with Unicode path support."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size > 0:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    try:
        with Image.open(path) as pil_img:
            return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Safely write a lossless PNG image with Unicode path support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def _unique_timestamped_name(parent: Path, stem: str, suffix: str = "") -> str:
    """Create YYYY-MM-DD_HH-MM-SS_<stem><suffix>, waiting 1s only on same-second collision."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = f"{ts}_{stem}{suffix}"
    if (parent / candidate).exists():
        time.sleep(1.05)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = f"{ts}_{stem}{suffix}"
    return candidate


def _get_manifest_dir(m2_res: dict) -> Path:
    artifacts_dir = m2_res.get("artifacts_directory", {})
    manifest_ref = artifacts_dir.get("manifest") or artifacts_dir.get("manifest_path")
    if manifest_ref:
        return Path(manifest_ref).parent
    case_id = m2_res.get("forensics_metadata", {}).get("case_id", "case")
    return Path(artifacts_dir.get("root", PROCESSED_ROOT / case_id))


def _save_module2_json_records(m2_res: dict) -> dict[int, str]:
    """Save per-frame Module 2 JSON records alongside preprocessing_manifest.json."""
    manifest_dir = _get_manifest_dir(m2_res)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = manifest_dir / "preprocessing_manifest.json"
    if not manifest_file.exists():
        manifest_file.write_text(
            json.dumps(m2_res, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    saved_paths: dict[int, str] = {}
    used_names: set[str] = set()
    for idx, frame_rec in enumerate(m2_res.get("frames_detail", [])):
        frame_stem = Path(frame_rec.get("frame_path", f"frame_{idx + 1:05d}")).stem
        candidate_name = f"{frame_stem}_forensic_record.json"
        if candidate_name in used_names:
            candidate_name = f"{frame_stem}_{idx + 1:05d}_forensic_record.json"
        used_names.add(candidate_name)

        record_path = manifest_dir / candidate_name
        record_path.write_text(
            json.dumps(frame_rec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_paths[idx] = str(record_path.resolve())

    return saved_paths


def _run_module3_on_raw_frames(m2_res: dict, ela_quality: int = 90) -> dict:
    """
    Execute Upgraded Module 3 on unedited raw frames (frame_path):
      1. NoiseAnalyzer (SRM residuals + face/background contrast + blending seam) — Video & Image.
      2. FFTAnalyzer (1/f decay + chroma FFT + periodic spikes, full frame + unwarped face crops).
      3. ELAAnalyzer — Included in score ONLY for native JPEG images (excluded for video PNGs).
    """
    manifest_dir = _get_manifest_dir(m2_res)
    noise_dir = manifest_dir / "noise"
    ela_dir = manifest_dir / "ela"
    fft_dir = manifest_dir / "fft"
    fft_crops_dir = fft_dir / "raw_crops"
    noise_dir.mkdir(parents=True, exist_ok=True)
    ela_dir.mkdir(parents=True, exist_ok=True)
    fft_dir.mkdir(parents=True, exist_ok=True)

    noise_analyzer = NoiseAnalyzer()
    ela_analyzer = ELAAnalyzer(quality=ela_quality)
    fft_analyzer = FFTAnalyzer()
    translator = Module3ForensicTranslator()

    frames_detail = m2_res.get("frames_detail", [])
    frame_results: list[dict] = []
    used_names: set[str] = set()

    for idx, frame_rec in enumerate(frames_detail):
        frame_path = frame_rec["frame_path"]
        frame_stem = Path(frame_path).stem
        faces = frame_rec.get("faces", []) or []
        face_bboxes = [f["bbox"] for f in faces if "bbox" in f]

        # 1. SRM Noise Residual & Boundary Blending Analysis (primary spatial check)
        noise_res = noise_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(noise_dir),
            face_bboxes=face_bboxes if face_bboxes else None,
            prefix=frame_stem,
        )

        # 2. ELA on raw frame (scored ONLY when source is native JPEG)
        ela_res = ela_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(ela_dir),
            face_bboxes=face_bboxes if face_bboxes else None,
            prefix=frame_stem,
        )

        # 3. Full-frame Upgraded 2D-FFT (auto-trims solid white/black outer margins)
        full_fft_res = fft_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(fft_dir),
            prefix=f"{frame_stem}_frame",
            save_radial_plot=True,
            trim_borders=True,
        )

        # 4. Unwarped raw face bbox 2D-FFT (direct pixel slice, zero affine distortion)
        face_raw_ffts: list[dict] = []
        if faces:
            raw_bgr = _imread_unicode(Path(frame_path))
            if raw_bgr is not None:
                img_h, img_w = raw_bgr.shape[:2]
                for f_info in faces:
                    bbox = f_info.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    x1 = max(0, int(np.floor(float(bbox[0]))))
                    y1 = max(0, int(np.floor(float(bbox[1]))))
                    x2 = min(img_w, int(np.ceil(float(bbox[2]))))
                    y2 = min(img_h, int(np.ceil(float(bbox[3]))))
                    if (x2 - x1) < 32 or (y2 - y1) < 32:
                        continue

                    face_idx = int(f_info.get("face_index", len(face_raw_ffts)))
                    crop_bgr = raw_bgr[y1:y2, x1:x2]
                    crop_path = fft_crops_dir / f"{frame_stem}_face_{face_idx:02d}_raw_crop.png"
                    if _imwrite_unicode(crop_path, crop_bgr):
                        crop_fft = fft_analyzer.analyze(
                            image_path=str(crop_path),
                            output_dir=str(fft_dir),
                            prefix=f"{frame_stem}_face_{face_idx:02d}",
                            save_radial_plot=True,
                            trim_borders=False,
                        )
                        face_raw_ffts.append(
                            {
                                "face_index": face_idx,
                                "global_face_id": f_info.get("global_face_id")
                                or f"Face #{face_idx}",
                                "bbox": [x1, y1, x2, y2],
                                "raw_crop_path": str(crop_path.resolve()),
                                **crop_fft,
                            }
                        )

        max_face_fft = (
            max(item["fft_score"] for item in face_raw_ffts)
            if face_raw_ffts
            else None
        )
        effective_fft_score = (
            round(max(full_fft_res["fft_score"], max_face_fft), 4)
            if max_face_fft is not None
            else full_fft_res["fft_score"]
        )

        # Peak-weighted fusion across active signals (ELA included ONLY for native JPEG)
        active_scores = [noise_res["noise_score"], effective_fft_score]
        ela_used = bool(ela_res.get("is_jpeg", False))
        if ela_used:
            active_scores.append(ela_res["ela_score"])

        combined_m3_score = round(
            float(0.60 * max(active_scores) + 0.40 * float(np.mean(active_scores))),
            4,
        )

        rec_name = f"{frame_stem}_module3_record.json"
        if rec_name in used_names:
            rec_name = f"{frame_stem}_{idx + 1:05d}_module3_record.json"
        used_names.add(rec_name)
        rec_path = manifest_dir / rec_name

        frame_m3_record = {
            "frame_index": idx + 1,
            "frame_path": frame_path,
            "noise": noise_res,
            "ela": ela_res,
            "fft_full_frame": full_fft_res,
            "fft_unwarped_faces": face_raw_ffts,
            "summary": {
                "noise_score": noise_res["noise_score"],
                "noise_suspicion": noise_res["suspicion_level"],
                "ela_score": ela_res["ela_score"],
                "ela_suspicion": ela_res["suspicion_level"],
                "ela_used_in_score": ela_used,
                "frame_fft_score": full_fft_res["fft_score"],
                "max_face_fft_score": max_face_fft,
                "effective_fft_score": effective_fft_score,
                "fft_suspicion": FFTAnalyzer._compute_suspicion_level(effective_fft_score),
                "combined_module3_score": combined_m3_score,
                "combined_suspicion": NoiseAnalyzer._compute_suspicion_level(combined_m3_score),
            },
            "record_json_path": str(rec_path.resolve()),
        }
        frame_m3_record["interpretation"] = translator.interpret_frame(frame_m3_record)

        rec_path.write_text(
            json.dumps(frame_m3_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        frame_results.append(frame_m3_record)

    noise_scores = [r["summary"]["noise_score"] for r in frame_results]
    ela_scores = [r["summary"]["ela_score"] for r in frame_results]
    fft_scores = [r["summary"]["effective_fft_score"] for r in frame_results]
    combined_scores = [r["summary"]["combined_module3_score"] for r in frame_results]
    any_ela_used = any(r["summary"]["ela_used_in_score"] for r in frame_results)

    mean_noise = round(float(np.mean(noise_scores)), 4) if noise_scores else 0.0
    max_noise = round(float(np.max(noise_scores)), 4) if noise_scores else 0.0
    mean_ela = round(float(np.mean(ela_scores)), 4) if ela_scores else 0.0
    max_ela = round(float(np.max(ela_scores)), 4) if ela_scores else 0.0
    mean_fft = round(float(np.mean(fft_scores)), 4) if fft_scores else 0.0
    max_fft = round(float(np.max(fft_scores)), 4) if fft_scores else 0.0
    mean_comb = round(float(np.mean(combined_scores)), 4) if combined_scores else 0.0
    max_comb = round(float(np.max(combined_scores)), 4) if combined_scores else 0.0

    m3_manifest_path = manifest_dir / "module3_manifest.json"
    m3_manifest = {
        "forensics_metadata": {
            "case_id": m2_res["forensics_metadata"]["case_id"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "media_type": m2_res["forensics_metadata"]["media_type"],
        },
        "module3_settings": {
            "input_source": "Unedited Raw Frames (frame_path)",
            "active_signals": (
                ["Noise & Boundary (SRM)", "Upgraded 2D-FFT (1/f + Chroma)", "JPEG ELA"]
                if any_ela_used
                else ["Noise & Boundary (SRM)", "Upgraded 2D-FFT (1/f + Chroma)"]
            ),
            "ela_quality": ela_quality,
            "ela_scored_on_native_jpeg_only": True,
            "fft_modes": ["Unwarped Raw Face BBox Crop", "Full Raw Frame"],
        },
        "execution_summary": {
            "frames_analyzed": len(frame_results),
            "mean_noise_score": mean_noise,
            "max_noise_score": max_noise,
            "mean_ela_score": mean_ela,
            "max_ela_score": max_ela,
            "ela_used_in_scoring": any_ela_used,
            "mean_fft_score": mean_fft,
            "max_fft_score": max_fft,
            "mean_module3_score": mean_comb,
            "max_module3_score": max_comb,
            "overall_suspicion": NoiseAnalyzer._compute_suspicion_level(max_comb),
        },
        "artifacts_directory": {
            "root": str(manifest_dir.resolve()),
            "noise": str(noise_dir.resolve()),
            "ela": str(ela_dir.resolve()),
            "fft": str(fft_dir.resolve()),
            "manifest": str(m3_manifest_path.resolve()),
        },
        "frames_detail": frame_results,
    }
    m3_manifest["interpretation"] = translator.interpret_manifest(m3_manifest)

    m3_manifest_path.write_text(
        json.dumps(m3_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return m3_manifest


def _suspicion_display(severity: str, score: float | None = None) -> tuple[str, str, str]:
    """Return (label, icon, css_class) for a given suspicion severity."""
    sev = str(severity).lower()
    pct_str = f" ({float(score) * 100:.0f}%)" if score is not None else ""
    if sev == "high":
        return f"HIGH{pct_str}", "🔴", "status-high"
    if sev == "medium":
        return f"MODERATE{pct_str}", "🟡", "status-medium"
    return f"LOW{pct_str}", "🟢", "status-low"


def _render_suspicion_box(
    title: str,
    severity: str,
    score: float | None = None,
    subtitle: str | None = None,
) -> None:
    """Render a high-visibility colored quick-glance status box."""
    label, icon, css_cls = _suspicion_display(severity, score)
    sub_html = f'<div class="status-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="status-banner {css_cls}">
            <div class="status-title">{icon} {title}: <span>{label}</span></div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_and_details(
    translation: dict | None,
    verdict_title: str = "Verdict",
    expander_label: str = "📖 Click to expand detailed plain-English explanation",
) -> None:
    """
    Show a short, colored verdict box first, followed by a collapsed expander
    containing simple plain-English details for non-technical users.
    """
    if not translation:
        return

    sev = str(translation.get("severity", "low")).lower()
    headline = translation.get("headline", "")

    if sev == "high":
        st.error(f"**{verdict_title}:** {headline}")
    elif sev == "medium":
        st.warning(f"**{verdict_title}:** {headline}")
    else:
        st.success(f"**{verdict_title}:** {headline}")

    findings = translation.get("findings", []) or []
    guidance = translation.get("user_guidance")
    if not findings and not guidance:
        return

    icon_map = {
        "alert": "🔴",
        "warning": "⚠️",
        "ok": "✅",
        "info": "ℹ️",
    }

    with st.expander(expander_label, expanded=False):
        for item in findings:
            item_icon = icon_map.get(item.get("severity", "info"), "ℹ️")
            st.markdown(
                f"{item_icon} **{item.get('title', 'Finding')}:** {item.get('detail', '')}"
            )
        if guidance:
            st.markdown("---")
            st.caption(f"💡 {guidance}")


def _render_frame_forensic_record(
    current_frame: dict, record_file_path: str | None = None
) -> None:
    """Parse and render a frame's Module 2 forensic record neatly without raw JSON."""
    st.markdown("##### 📄 Frame-Level Forensic Record")

    frame_path = current_frame.get("frame_path", "")
    frame_name = Path(frame_path).name if frame_path else "N/A"
    resolution = current_frame.get("image_resolution", [])
    res_str = (
        f"{resolution[0]} x {resolution[1]} px"
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2
        else "N/A"
    )
    faces = current_frame.get("faces", []) or []
    faces_detected = current_frame.get("faces_detected", len(faces))
    primary_detector = (
        str(faces[0].get("detector", "N/A")).upper() if faces else "N/A"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frame File", frame_name)
    m2.metric("Resolution", res_str)
    m3.metric("Faces Detected", faces_detected)
    m4.metric("AI Detector", primary_detector)

    known_keys = {
        "frame_path",
        "image_resolution",
        "faces_detected",
        "annotated_frame_path",
        "faces",
    }
    meta_props = [
        "Source Frame File",
        "Source Frame Path",
        "Annotated Overlay Path",
        "Saved Record (.json)",
    ]
    meta_vals = [
        frame_name,
        frame_path or "N/A",
        current_frame.get("annotated_frame_path") or "N/A",
        record_file_path or "N/A",
    ]
    for extra_key, extra_val in current_frame.items():
        if extra_key not in known_keys and not isinstance(extra_val, (dict, list)):
            meta_props.append(extra_key.replace("_", " ").title())
            meta_vals.append(str(extra_val))

    st.markdown("###### 🗂️ Frame & Artifact Metadata")
    st.table({"Property": meta_props, "Value": meta_vals})

    st.markdown("###### 👤 Detected Faces Breakdown")
    if not faces:
        st.info(
            "No face met the detection threshold or geometric sanity check for this frame."
        )
        return

    summary_rows = {
        "Face ID": [],
        "Detector": [],
        "Confidence": [],
        "Bounding Box [X1, Y1, X2, Y2]": [],
        "Box Size (W x H)": [],
        "Aligned Size": [],
    }
    for f_info in faces:
        bbox = f_info.get("bbox", [0, 0, 0, 0])
        box_w = round(float(bbox[2]) - float(bbox[0]), 2) if len(bbox) >= 4 else 0.0
        box_h = round(float(bbox[3]) - float(bbox[1]), 2) if len(bbox) >= 4 else 0.0
        aligned_sz = f_info.get("aligned_size", [])
        aligned_str = (
            f"{aligned_sz[0]} x {aligned_sz[1]} px"
            if isinstance(aligned_sz, (list, tuple)) and len(aligned_sz) >= 2
            else "N/A"
        )
        face_label = f_info.get("global_face_id") or f"Face #{f_info.get('face_index', 0)}"

        summary_rows["Face ID"].append(face_label)
        summary_rows["Detector"].append(str(f_info.get("detector", "N/A")).upper())
        summary_rows["Confidence"].append(f"{float(f_info.get('confidence', 0.0)) * 100:.2f}%")
        summary_rows["Bounding Box [X1, Y1, X2, Y2]"].append(
            f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if len(bbox) >= 4 else "N/A"
        )
        summary_rows["Box Size (W x H)"].append(f"{box_w} x {box_h} px")
        summary_rows["Aligned Size"].append(aligned_str)

    st.table(summary_rows)

    landmark_labels = [
        ("left_eye", "Left Eye"),
        ("right_eye", "Right Eye"),
        ("nose", "Nose"),
        ("mouth_left", "Mouth Left"),
        ("mouth_right", "Mouth Right"),
    ]

    for idx, f_info in enumerate(faces):
        face_label = f_info.get("global_face_id") or f"Face #{f_info.get('face_index', idx)}"
        conf_pct = float(f_info.get("confidence", 0.0)) * 100.0
        detector_name = str(f_info.get("detector", "N/A")).upper()

        with st.container(border=True):
            st.markdown(
                f"**{face_label}** — Detector: `{detector_name}` | Confidence: `{conf_pct:.2f}%`"
            )
            col_geo, col_lm = st.columns([1.2, 1])

            bbox = f_info.get("bbox", [0, 0, 0, 0])
            box_w = round(float(bbox[2]) - float(bbox[0]), 2) if len(bbox) >= 4 else 0.0
            box_h = round(float(bbox[3]) - float(bbox[1]), 2) if len(bbox) >= 4 else 0.0
            aligned_sz = f_info.get("aligned_size", [])
            aligned_str = (
                f"{aligned_sz[0]} x {aligned_sz[1]} px"
                if isinstance(aligned_sz, (list, tuple)) and len(aligned_sz) >= 2
                else "N/A"
            )
            aligned_path = f_info.get("aligned_face_path", "")

            with col_geo:
                st.caption("Bounding Box & Alignment")
                st.table(
                    {
                        "Metric": [
                            "Face Index",
                            "Top-Left (X1, Y1)",
                            "Bottom-Right (X2, Y2)",
                            "Box Dimensions (W x H)",
                            "Canonical Output Size",
                            "Aligned Face File",
                        ],
                        "Value": [
                            str(f_info.get("face_index", idx)),
                            f"({bbox[0]}, {bbox[1]})" if len(bbox) >= 4 else "N/A",
                            f"({bbox[2]}, {bbox[3]})" if len(bbox) >= 4 else "N/A",
                            f"{box_w} x {box_h} px",
                            aligned_str,
                            Path(aligned_path).name if aligned_path else "N/A",
                        ],
                    }
                )

            with col_lm:
                st.caption("5-Point Facial Landmarks (px)")
                landmarks = f_info.get("landmarks", {}) or {}
                lm_names, lm_x, lm_y = [], [], []
                for key, label in landmark_labels:
                    pt = landmarks.get(key)
                    lm_names.append(label)
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        lm_x.append(str(pt[0]))
                        lm_y.append(str(pt[1]))
                    else:
                        lm_x.append("N/A")
                        lm_y.append("N/A")
                st.table({"Landmark": lm_names, "X": lm_x, "Y": lm_y})


def _render_module3_noise(noise_res: dict, noise_translation: dict | None = None) -> None:
    """Render Module 3.1 Noise & Boundary Analysis: Visuals -> Verdict -> Details -> Tucked Metrics."""
    st.markdown("##### 🔬 SRM Noise Residual & Face-Boundary Analysis")
    st.caption(
        "Works on both video frames and images. Strips away colors and lighting to expose unnatural "
        "smoothness, mismatched sensor grain, or face-swap blending seams."
    )

    c1, c2 = st.columns(2)
    with c1:
        res_p = noise_res.get("noise_residual_path")
        if res_p and os.path.exists(res_p):
            st.image(res_p, caption="High-Pass Pixel Grain (SRM Residual)", use_container_width=True)
    with c2:
        heat_p = noise_res.get("noise_heatmap_path")
        if heat_p and os.path.exists(heat_p):
            st.image(
                heat_p,
                caption="Local Noise Variance Heatmap (Blue = Smooth, Yellow/Red = High Grain)",
                use_container_width=True,
            )

    st.write("")

    _render_verdict_and_details(
        noise_translation,
        verdict_title="Noise & Boundary Verdict",
        expander_label="📖 Detailed Plain-English Noise Explanation",
    )

    m = noise_res.get("metrics", {}) or {}
    with st.expander("📊 Technical Noise & Boundary Metrics (For Forensic Reviewers)", expanded=False):
        st.table(
            {
                "Noise / Boundary Metric": [
                    "Noise Suspicion Score",
                    "Suspicion Level",
                    "Noise P95 / P50 Ratio",
                    "Block Noise CV",
                    "Global Noise Median",
                    "Face Noise Median",
                    "Background Noise Median",
                    "Face-vs-Background Noise Contrast",
                    "Face Smoother Than Background",
                    "Face / BG Chrominance Noise",
                    "Face-vs-BG Chroma Contrast",
                    "Boundary Seam Transition Ratio",
                ],
                "Observed Value": [
                    f"{noise_res['noise_score']:.4f} ({noise_res['noise_score'] * 100:.0f}%)",
                    str(noise_res["suspicion_level"]).upper(),
                    str(m.get("noise_p95_p50_ratio")),
                    str(m.get("noise_block_cv")),
                    str(m.get("global_noise_median")),
                    str(m.get("face_noise_median")) if m.get("face_noise_median") is not None else "N/A",
                    str(m.get("bg_noise_median")) if m.get("bg_noise_median") is not None else "N/A",
                    str(m.get("face_bg_noise_contrast")) if m.get("face_bg_noise_contrast") is not None else "N/A",
                    str(m.get("face_smoother_than_bg")) if m.get("face_smoother_than_bg") is not None else "N/A",
                    f"{m.get('face_chroma_noise')} / {m.get('bg_chroma_noise')}"
                    if m.get("face_chroma_noise") is not None
                    else "N/A",
                    str(m.get("face_bg_chroma_contrast")) if m.get("face_bg_chroma_contrast") is not None else "N/A",
                    str(m.get("boundary_seam_ratio")) if m.get("boundary_seam_ratio") is not None else "N/A",
                ],
            }
        )


def _render_module3_ela(ela_res: dict, ela_translation: dict | None = None) -> None:
    """Render Module 3.3 ELA (scored only for native JPEG): Visuals -> Verdict -> Details -> Tucked Metrics."""
    st.markdown("##### 🔥 Error Level Analysis (ELA — Native JPEG Specialist)")
    if not ela_res.get("is_jpeg"):
        st.info(
            f"ℹ️ This frame is **{ela_res.get('original_format', 'PNG')}** (not a native JPEG photo). "
            "ELA visuals are shown for reference, **excluded from Module 3 suspicion scoring** so video frames are not diluted."
        )
    else:
        st.caption(
            "Re-saves the JPEG to highlight areas that react differently to compression. "
            "Brighter pixels indicate larger changes."
        )

    c1, c2 = st.columns(2)
    with c1:
        raw_p = ela_res.get("ela_raw_path")
        if raw_p and os.path.exists(raw_p):
            st.image(raw_p, caption="Amplified Compression Differences (RGB)", use_container_width=True)
    with c2:
        heat_p = ela_res.get("ela_heatmap_path")
        if heat_p and os.path.exists(heat_p):
            st.image(heat_p, caption="Compression Heatmap (Brighter = Higher Difference)", use_container_width=True)

    st.write("")

    _render_verdict_and_details(
        ela_translation,
        verdict_title="ELA Verdict",
        expander_label="📖 Detailed Plain-English ELA Explanation",
    )

    metrics = ela_res.get("metrics", {})
    face_mean = metrics.get("face_region_mean")
    bg_mean = metrics.get("bg_region_mean")
    contrast = metrics.get("face_bg_contrast")

    with st.expander("📊 Technical ELA Metrics (For Forensic Reviewers)", expanded=False):
        st.caption(ela_res.get("note", ""))
        st.table(
            {
                "ELA Metric": [
                    "ELA Suspicion Score",
                    "Included in Module 3 Score",
                    "Original Format",
                    "Re-compression Quality",
                    "Mean / Std / Max Error",
                    "Block P95 / P50 Ratio",
                    "Block Coefficient of Variation (CV)",
                    "Hotspot Pixel Ratio",
                    "Face Region Mean Error",
                    "Background Region Mean Error",
                    "Face-vs-Background Contrast",
                ],
                "Observed Value": [
                    f"{ela_res['ela_score']:.4f} ({ela_res['ela_score'] * 100:.0f}%)",
                    "Yes (Native JPEG)" if ela_res.get("is_jpeg") else "No (Excluded for Non-JPEG/Video)",
                    str(ela_res.get("original_format", "N/A")),
                    f"JPEG Q={ela_res.get('ela_quality', 90)}",
                    f"{metrics.get('mean_error')} / {metrics.get('std_error')} / {metrics.get('max_error')}",
                    str(metrics.get("block_p95_p50_ratio")),
                    str(metrics.get("block_cv")),
                    str(metrics.get("hotspot_ratio")),
                    str(face_mean) if face_mean is not None else "N/A (No face bbox)",
                    str(bg_mean) if bg_mean is not None else "N/A (No face bbox)",
                    str(contrast) if contrast is not None else "N/A (No face bbox)",
                ],
            }
        )


def _render_module3_fft(frame_m3: dict) -> None:
    """Render Module 3.2 Upgraded 2D-FFT: Unwarped Face Crops First -> Full Frame -> Verdict -> Details -> Tucked Metrics."""
    full_fft = frame_m3["fft_full_frame"]
    face_ffts = frame_m3.get("fft_unwarped_faces", [])
    fft_translation = frame_m3.get("interpretation", {}).get("fft")

    # 1A. Unwarped Face Crops First
    st.markdown("##### 🎯 Unwarped Face Region Frequency Scan (Primary Check)")
    st.caption(
        "Scans each detected face directly from the raw image (without stretching or warping) "
        "for repeating AI grid traces, chrominance anomalies, and 1/f detail curve deviations."
    )

    if not face_ffts:
        st.info(
            "No valid face regions (>= 32x32 px) were detected in this frame; see the full-frame scan below."
        )
    else:
        for f_fft in face_ffts:
            label, icon, _ = _suspicion_display(
                f_fft["suspicion_level"], f_fft["fft_score"]
            )
            with st.container(border=True):
                st.markdown(
                    f"**{f_fft['global_face_id']}** — Frequency Suspicion: {icon} `{label}`"
                )
                col_crop, col_spec, col_rad = st.columns([0.8, 1.1, 1.1])
                with col_crop:
                    if os.path.exists(f_fft["raw_crop_path"]):
                        st.image(
                            f_fft["raw_crop_path"],
                            caption=f"Raw Face Slice ({f_fft['image_resolution'][0]}x{f_fft['image_resolution'][1]} px)",
                            use_container_width=True,
                        )
                with col_spec:
                    if f_fft.get("fft_spectrum_path") and os.path.exists(f_fft["fft_spectrum_path"]):
                        st.image(
                            f_fft["fft_spectrum_path"],
                            caption="Face Frequency Pattern (2D-FFT)",
                            use_container_width=True,
                        )
                with col_rad:
                    if f_fft.get("fft_radial_plot_path") and os.path.exists(f_fft["fft_radial_plot_path"]):
                        st.image(
                            f_fft["fft_radial_plot_path"],
                            caption="Face Detail Drop-Off vs. 1/f Camera Trend",
                            use_container_width=True,
                        )

    # 1B. Full Raw Frame 2D-FFT Second
    st.markdown("##### 📡 Full Frame Frequency Scan (Border-Trimmed)")
    c1, c2 = st.columns(2)
    with c1:
        spec_p = full_fft.get("fft_spectrum_path")
        if spec_p and os.path.exists(spec_p):
            st.image(spec_p, caption="Full-Frame Frequency Pattern (2D-FFT)", use_container_width=True)
    with c2:
        rad_p = full_fft.get("fft_radial_plot_path")
        if rad_p and os.path.exists(rad_p):
            st.image(rad_p, caption="Full-Frame Detail Drop-Off vs. 1/f Camera Trend", use_container_width=True)

    st.write("")

    _render_verdict_and_details(
        fft_translation,
        verdict_title="2D-FFT Verdict",
        expander_label="📖 Detailed Plain-English 2D-FFT Explanation",
    )

    with st.expander("📊 Technical 2D-FFT Metrics (For Forensic Reviewers)", expanded=False):
        if face_ffts:
            for f_fft in face_ffts:
                fm = f_fft.get("metrics", {})
                st.markdown(
                    f"**{f_fft['global_face_id']}** — BBox `{f_fft['bbox']}` | "
                    f"Score: `{f_fft['fft_score']:.4f} ({f_fft['fft_score'] * 100:.0f}%)` "
                    f"(`{str(f_fft['suspicion_level']).upper()}`)"
                )
                st.table(
                    {
                        "Face-ROI FFT Metric": [
                            "Luminance HF Energy Ratio",
                            "Chrominance (Cb/Cr) HF Ratio",
                            "Max Periodic Spike Z-Score",
                            "Top-20 Spike Mean Z-Score",
                            "Spike Density (>4σ)",
                            "Radial Curve Roughness",
                            "1/f Decay Fit RMSE",
                            "Spectral Slope (alpha)",
                        ],
                        "Observed Value": [
                            str(fm.get("hf_energy_ratio")),
                            str(fm.get("chroma_hf_ratio")),
                            str(fm.get("max_spike_z")),
                            str(fm.get("top20_spike_mean_z")),
                            str(fm.get("spike_density")),
                            str(fm.get("radial_roughness")),
                            str(fm.get("radial_fit_rmse")),
                            str(fm.get("spectral_slope")),
                        ],
                    }
                )

        f_metrics = full_fft.get("metrics", {})
        st.markdown("**Full Raw Frame 2D-FFT Metrics**")
        st.table(
            {
                "Full-Frame FFT Metric": [
                    "Full-Frame FFT Score",
                    "Suspicion Level",
                    "Active Content Resolution",
                    "Luminance HF Energy Ratio",
                    "Chrominance (Cb/Cr) HF Ratio",
                    "Max Periodic Spike Z-Score",
                    "Top-20 Spike Mean Z-Score",
                    "Spike Density (>4σ)",
                    "Radial Curve Roughness",
                    "1/f Decay Fit RMSE",
                    "Spectral Slope (alpha)",
                ],
                "Observed Value": [
                    f"{full_fft['fft_score']:.4f} ({full_fft['fft_score'] * 100:.0f}%)",
                    str(full_fft["suspicion_level"]).upper(),
                    f"{full_fft['image_resolution'][0]} x {full_fft['image_resolution'][1]} px",
                    str(f_metrics.get("hf_energy_ratio")),
                    str(f_metrics.get("chroma_hf_ratio")),
                    str(f_metrics.get("max_spike_z")),
                    str(f_metrics.get("top20_spike_mean_z")),
                    str(f_metrics.get("spike_density")),
                    str(f_metrics.get("radial_roughness")),
                    str(f_metrics.get("radial_fit_rmse")),
                    str(f_metrics.get("spectral_slope")),
                ],
            }
        )


def _render_module3_frame_record(frame_m3: dict) -> None:
    """Render frame-level Module 3 verdict, plain-English takeaways, and tucked technical record."""
    st.markdown("##### 📄 Frame-Level Module 3 Summary")

    interp = frame_m3.get("interpretation", {}) or {}
    sev = str(interp.get("severity", "low")).lower()
    headline = interp.get("headline", "")

    if headline:
        if sev == "high":
            st.error(f"**Frame Verdict:** {headline}")
        elif sev == "medium":
            st.warning(f"**Frame Verdict:** {headline}")
        else:
            st.success(f"**Frame Verdict:** {headline}")

    takeaways = interp.get("key_takeaways", []) or []
    if takeaways:
        with st.expander("📖 Detailed Frame Breakdown (Noise, FFT & ELA)", expanded=False):
            for takeaway in takeaways:
                st.markdown(f"- {takeaway}")

    s = frame_m3["summary"]
    with st.expander("📊 Technical Frame Record & File Paths", expanded=False):
        st.table(
            {
                "Module 3 Property": [
                    "Frame Index",
                    "Source Raw Frame",
                    "Noise & Boundary Score",
                    "Full-Frame FFT Score",
                    "Max Unwarped Face FFT Score",
                    "Effective FFT Score",
                    "ELA Score (Included in Score?)",
                    "Combined Module 3 Score",
                    "Saved Module 3 Record (.json)",
                ],
                "Observed Value": [
                    str(frame_m3["frame_index"]),
                    frame_m3["frame_path"],
                    f"{s['noise_score']:.4f} ({s['noise_score'] * 100:.0f}% — {str(s['noise_suspicion']).upper()})",
                    f"{s['frame_fft_score']:.4f} ({s['frame_fft_score'] * 100:.0f}%)",
                    f"{s['max_face_fft_score']:.4f} ({s['max_face_fft_score'] * 100:.0f}%)"
                    if s["max_face_fft_score"] is not None
                    else "N/A",
                    f"{s['effective_fft_score']:.4f} ({s['effective_fft_score'] * 100:.0f}% — {str(s['fft_suspicion']).upper()})",
                    f"{s['ela_score']:.4f} ({'Yes' if s.get('ela_used_in_score') else 'No — Non-JPEG'})",
                    f"{s['combined_module3_score']:.4f} ({s['combined_module3_score'] * 100:.0f}% — {str(s['combined_suspicion']).upper()})",
                    frame_m3["record_json_path"],
                ],
            }
        )


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
    .status-banner {
        border-radius: 10px;
        padding: 12px 18px;
        margin: 8px 0 14px 0;
        border-left: 6px solid;
    }
    .status-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .status-sub {
        font-size: 0.90rem;
        opacity: 0.9;
        margin-top: 4px;
    }
    .status-low {
        background-color: rgba(33, 195, 84, 0.14);
        border-color: #21c354;
    }
    .status-medium {
        background-color: rgba(255, 189, 46, 0.16);
        border-color: #ffbd2e;
    }
    .status-high {
        background-color: rgba(255, 75, 75, 0.16);
        border-color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "settings_locked" not in st.session_state:
    st.session_state["settings_locked"] = False
if "applied_settings" not in st.session_state:
    st.session_state["applied_settings"] = {
        "detector_backend": "retinaface",
        "min_confidence": 0.80,
        "output_size": 299,
        "max_frames": 10,
        "ela_quality": 90,
    }

st.title("🔍 Multimodal Explainable Forensics Framework")
st.caption(
    "End-to-End Deepfake and Digital Media Manipulation Forensics Pipeline with High Explainability (XAI)."
)

# =========================================================
# SIDEBAR: CONFIGURATION & INTEGRITY LOCK
# =========================================================
st.sidebar.header("⚙️ 1. Module 2 & 3 Settings")

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

sel_ela_quality = st.sidebar.slider(
    "Module 3 ELA JPEG Quality (Native JPEGs)",
    min_value=50,
    max_value=99,
    value=int(current_cfg.get("ela_quality", 90)),
    step=1,
    disabled=is_locked,
    help="Re-compression JPEG quality used by Error Level Analysis (ELA) on native JPEG images.",
)

st.sidebar.markdown("---")
if not is_locked:
    if st.sidebar.button(
        "🔒 Apply Settings (Lock)", use_container_width=True, type="primary"
    ):
        st.session_state["applied_settings"] = {
            "detector_backend": sel_backend,
            "min_confidence": sel_confidence,
            "output_size": sel_size,
            "max_frames": sel_max_frames,
            "ela_quality": sel_ela_quality,
        }
        st.session_state["settings_locked"] = True
        st.session_state.pop("m2_cache_key", None)
        st.rerun()
else:
    st.sidebar.success("🔒 Configuration is LOCKED for evidence integrity.")
    if st.sidebar.button("🔓 Unlock & Edit Settings", use_container_width=True):
        st.session_state["settings_locked"] = False
        st.session_state["m2_result"] = None
        st.session_state["m3_result"] = None
        st.session_state.pop("m2_cache_key", None)
        st.session_state.pop("m2_frame_record_paths", None)
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
    raw_bytes = uploaded_file.getbuffer()
    upload_buf_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
    upload_sig = getattr(uploaded_file, "file_id", None) or (
        f"{uploaded_file.name}_{uploaded_file.size}_{upload_buf_hash}"
    )

    existing_raw_path = st.session_state.get("active_raw_path")
    if (
        st.session_state.get("active_upload_sig") != upload_sig
        or not existing_raw_path
        or not os.path.exists(existing_raw_path)
    ):
        orig_p = Path(uploaded_file.name)
        raw_filename = _unique_timestamped_name(TEMP_DIR, orig_p.stem, orig_p.suffix)
        raw_dest = TEMP_DIR / raw_filename
        with open(raw_dest, "wb") as f:
            f.write(raw_bytes)
        st.session_state["active_upload_sig"] = upload_sig
        st.session_state["active_raw_path"] = str(raw_dest)
        st.session_state.pop("m2_cache_key", None)
        st.session_state["m2_result"] = None
        st.session_state["m3_result"] = None

    file_path = st.session_state["active_raw_path"]
    original_stem = Path(uploaded_file.name).stem

    # =====================================================
    # MODULE 1: EVIDENCE INGESTION & DATA INTEGRITY
    # =====================================================
    st.header("🟢 Module 1: Evidence Ingestion & Chain of Custody")
    st.caption(f"🗄️ Working Evidence Copy: `{file_path}`")
    col_ev1, col_ev2 = st.columns([1, 1])

    with col_ev1:
        st.subheader("Raw Evidence")
        if uploaded_file.name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            col_l, col_img, col_r = st.columns([0.1, 0.8, 0.1])
            with col_img:
                st.image(Image.open(file_path), caption=Path(file_path).name, width=420)
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
    # MODULE 2 & MODULE 3 EXECUTION (LOCKED)
    # =====================================================
    st.divider()
    st.header("🟡 Module 2: AI Preprocessing & ROI Extraction")

    if not is_locked:
        st.warning(
            "⚠️ **Execution Paused:** Configuration is currently **UNLOCKED**. "
            "Please configure the parameters on the sidebar, then click **'🔒 Apply Settings (Lock)'** to run Module 2 & Module 3."
        )
    else:
        active_cfg = st.session_state["applied_settings"]
        file_sha = hashes["SHA-256"][:8] if hashes else "nohash"
        cache_key = (
            f"{file_sha}_{active_cfg['detector_backend']}_{active_cfg['min_confidence']:.2f}_"
            f"{active_cfg['output_size']}_{active_cfg['max_frames']}_{active_cfg['ela_quality']}_m3v2"
        )

        if st.session_state.get("m2_cache_key") != cache_key:
            with st.spinner(
                f"Executing Module 2 ({active_cfg['detector_backend'].upper()}) & Module 3 (Noise, Upgraded 2D-FFT, ELA)..."
            ):
                try:
                    short_case_id = _unique_timestamped_name(PROCESSED_ROOT, original_stem)
                    m2_res = run_module2(
                        input_path=file_path,
                        processed_root=str(PROCESSED_ROOT),
                        case_id=short_case_id,
                        output_size=(active_cfg["output_size"], active_cfg["output_size"]),
                        max_frames=active_cfg["max_frames"],
                        min_confidence=active_cfg["min_confidence"],
                        backend=active_cfg["detector_backend"],
                    )
                    frame_record_paths = _save_module2_json_records(m2_res)
                    m3_res = _run_module3_on_raw_frames(
                        m2_res, ela_quality=active_cfg["ela_quality"]
                    )
                    st.session_state["m2_cache_key"] = cache_key
                    st.session_state["m2_result"] = m2_res
                    st.session_state["m2_frame_record_paths"] = frame_record_paths
                    st.session_state["m3_result"] = m3_res
                    st.session_state["m2_error"] = None
                except Exception as exc:
                    st.session_state["m2_cache_key"] = cache_key
                    st.session_state["m2_result"] = None
                    st.session_state["m2_frame_record_paths"] = {}
                    st.session_state["m3_result"] = None
                    st.session_state["m2_error"] = str(exc)

        m2_res = st.session_state.get("m2_result")
        m3_res = st.session_state.get("m3_result")
        m2_err = st.session_state.get("m2_error")
        frame_record_paths = st.session_state.get("m2_frame_record_paths", {})

        if m2_err:
            st.error(f"Pipeline Error: {m2_err}")
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
            # MODULE 2 INTERACTIVE FRAME INSPECTION
            # =================================================
            total_frames = len(m2_res["frames_detail"])
            if total_frames > 0:
                st.markdown("---")
                if total_frames > 1:
                    col_sel1, col_sel2 = st.columns([2, 5])
                    with col_sel1:
                        st.markdown("#### 🎞️ Select Target Frame (Module 2)")
                    with col_sel2:
                        selected_frame_idx = st.slider(
                            "Target Frame (Module 2)",
                            min_value=1,
                            max_value=total_frames,
                            value=1,
                            step=1,
                            format="Frame #%d",
                            key="m2_frame_slider",
                            label_visibility="collapsed",
                        )
                else:
                    selected_frame_idx = 1

                current_frame = m2_res["frames_detail"][selected_frame_idx - 1]

                mode_options = [
                    "👤 1. Aligned Face ROI",
                    "🎯 2. BBox & Landmarks",
                    "🎞️ 3. Raw Keyframe",
                    "📄 4. Frame Forensic Record",
                ]

                st.write("")
                if hasattr(st, "segmented_control"):
                    selected_mode = st.segmented_control(
                        "Module 2 Inspection Mode:",
                        options=mode_options,
                        default=mode_options[0],
                        key="m2_mode_selector",
                        label_visibility="collapsed",
                    )
                else:
                    selected_mode = st.radio(
                        "Module 2 Inspection Mode:",
                        options=mode_options,
                        index=0,
                        horizontal=True,
                        key="m2_mode_selector",
                        label_visibility="collapsed",
                    )

                st.write("")
                st.info(
                    f"**Active Frame:** #{selected_frame_idx} (`{Path(current_frame['frame_path']).name}`) | "
                    f"**Detected Face ROIs:** {current_frame['faces_detected']}"
                )

                if selected_mode == mode_options[0]:
                    faces = current_frame.get("faces", [])
                    if not faces:
                        st.warning(
                            "No face met the detection threshold or geometric sanity check for this frame."
                        )
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
                    _render_frame_forensic_record(
                        current_frame,
                        record_file_path=frame_record_paths.get(selected_frame_idx - 1),
                    )

            # =================================================
            # MODULE 2 MANIFEST SUMMARY & AUDIT TRAIL
            # =================================================
            st.divider()
            st.subheader("📋 Preprocessing Manifest & Audit Summary")
            c_sum1, c_sum2 = st.columns([2, 1])
            with c_sum1:
                st.table(
                    {
                        "Forensic Property": [
                            "Case Identifier",
                            "Audit Timestamp",
                            "Media Type",
                            "AI Detector",
                            "Sampling Mode",
                            "Canonical Resolution",
                            "Sampled Frames",
                            "Blank / Fade Frames Filtered",
                            "Total Faces Extracted",
                            "Average Faces / Frame",
                            "Artifacts Directory",
                        ],
                        "Observed Value": [
                            m2_res["forensics_metadata"]["case_id"],
                            m2_res["forensics_metadata"]["timestamp"],
                            m2_res["forensics_metadata"]["media_type"],
                            m2_res["preprocessing_settings"]["ai_detector"],
                            m2_res["preprocessing_settings"]["sampling_mode"],
                            m2_res["preprocessing_settings"]["canonical_resolution"],
                            str(exec_sum["final_frames_analyzed"]),
                            str(exec_sum["discarded_blank_or_extreme_frames"]),
                            str(exec_sum["total_faces_extracted"]),
                            f"{exec_sum['average_faces_per_frame']} faces/frame",
                            m2_res["artifacts_directory"]["root"],
                        ],
                    }
                )
            with c_sum2:
                manifest_json_str = json.dumps(m2_res, ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Download preprocessing_manifest.json",
                    data=manifest_json_str,
                    file_name=f"{m2_res['forensics_metadata']['case_id']}_preprocessing_manifest.json",
                    mime="application/json",
                    type="primary",
                )

            # =================================================
            # MODULE 3: SIGNAL-LEVEL FORENSICS (NOISE, FFT, ELA)
            # =================================================
            if m3_res:
                st.divider()
                st.header("🟠 Module 3: Signal-Level Forensics (Noise, 2D-FFT & JPEG ELA)")
                st.caption(
                    "Analyzes unedited raw frames for pixel noise & face-swap blending seams (images & videos), "
                    "spectral/frequency anomalies (2D-FFT), and double-compression history (native JPEGs)."
                )

                m3_sum = m3_res["execution_summary"]

                _render_suspicion_box(
                    title="Module 3 Suspicion",
                    severity=m3_sum["overall_suspicion"],
                    score=m3_sum["max_module3_score"],
                    subtitle="Quick glance across analyzed frame(s) — inspect visuals below or scroll to the Audit Summary for the full verdict.",
                )

                mk1, mk2, mk3, mk4 = st.columns(4)
                mk1.metric("Frames Analyzed", m3_sum["frames_analyzed"])
                mk2.metric(
                    "Noise & Boundary (Mean / Peak)",
                    f"{m3_sum['mean_noise_score'] * 100:.0f}% / {m3_sum['max_noise_score'] * 100:.0f}%",
                )
                mk3.metric(
                    "2D-FFT Spectrum (Mean / Peak)",
                    f"{m3_sum['mean_fft_score'] * 100:.0f}% / {m3_sum['max_fft_score'] * 100:.0f}%",
                )
                mk4.metric(
                    "JPEG ELA (Mean / Peak)" if m3_sum.get("ela_used_in_scoring") else "JPEG ELA (Video/PNG)",
                    f"{m3_sum['mean_ela_score'] * 100:.0f}% / {m3_sum['max_ela_score'] * 100:.0f}%"
                    if m3_sum.get("ela_used_in_scoring")
                    else "Excluded (Non-JPEG)",
                )

                total_m3_frames = len(m3_res["frames_detail"])
                if total_m3_frames > 0:
                    st.markdown("---")
                    if total_m3_frames > 1:
                        m3_col1, m3_col2 = st.columns([2, 5])
                        with m3_col1:
                            st.markdown("#### 🎞️ Select Target Frame (Module 3)")
                        with m3_col2:
                            m3_frame_idx = st.slider(
                                "Target Frame (Module 3)",
                                min_value=1,
                                max_value=total_m3_frames,
                                value=1,
                                step=1,
                                format="Frame #%d",
                                key="m3_frame_slider",
                                label_visibility="collapsed",
                            )
                    else:
                        m3_frame_idx = 1

                    current_m3 = m3_res["frames_detail"][m3_frame_idx - 1]
                    m3_modes = [
                        "🔬 1. Noise & Boundary Analysis",
                        "📡 2. 2D-FFT Spectrum (Face & Frame)",
                        "🔥 3. Error Level Analysis (ELA)",
                        "📄 4. Frame Module 3 Summary",
                    ]

                    st.write("")
                    if hasattr(st, "segmented_control"):
                        m3_mode = st.segmented_control(
                            "Module 3 Inspection Mode:",
                            options=m3_modes,
                            default=m3_modes[0],
                            key="m3_mode_selector",
                            label_visibility="collapsed",
                        )
                    else:
                        m3_mode = st.radio(
                            "Module 3 Inspection Mode:",
                            options=m3_modes,
                            index=0,
                            horizontal=True,
                            key="m3_mode_selector",
                            label_visibility="collapsed",
                        )

                    cur_s = current_m3["summary"]
                    noise_lbl, _, _ = _suspicion_display(
                        cur_s["noise_suspicion"], cur_s["noise_score"]
                    )
                    fft_lbl, _, _ = _suspicion_display(
                        cur_s["fft_suspicion"], cur_s["effective_fft_score"]
                    )
                    ela_lbl = (
                        _suspicion_display(cur_s["ela_suspicion"], cur_s["ela_score"])[0]
                        if cur_s.get("ela_used_in_score")
                        else "N/A (Non-JPEG)"
                    )
                    _render_suspicion_box(
                        title=f"Frame #{m3_frame_idx} Status",
                        severity=cur_s["combined_suspicion"],
                        score=cur_s["combined_module3_score"],
                        subtitle=(
                            f"File: {Path(current_m3['frame_path']).name} | "
                            f"Noise: {noise_lbl} | 2D-FFT: {fft_lbl} | ELA: {ela_lbl}"
                        ),
                    )

                    if m3_mode == m3_modes[0]:
                        _render_module3_noise(
                            current_m3["noise"],
                            noise_translation=current_m3.get("interpretation", {}).get("noise"),
                        )
                    elif m3_mode == m3_modes[1]:
                        _render_module3_fft(current_m3)
                    elif m3_mode == m3_modes[2]:
                        _render_module3_ela(
                            current_m3["ela"],
                            ela_translation=current_m3.get("interpretation", {}).get("ela"),
                        )
                    elif m3_mode == m3_modes[3]:
                        _render_module3_frame_record(current_m3)

                # =============================================
                # MODULE 3 AUDIT SUMMARY & OVERALL VERDICT
                # =============================================
                st.divider()
                st.subheader("📋 Module 3 Forensic Audit Summary")

                _render_verdict_and_details(
                    m3_res.get("interpretation"),
                    verdict_title="Overall Module 3 Verdict",
                    expander_label="📖 Detailed Multi-Frame Breakdown & Limitations",
                )

                m3_c1, m3_c2 = st.columns([2, 1])
                with m3_c1:
                    with st.expander("🗂️ Technical Audit Metadata & Manifest Path", expanded=False):
                        st.table(
                            {
                                "Module 3 Property": [
                                    "Case Identifier",
                                    "Audit Timestamp",
                                    "Active Signals",
                                    "Mean / Max Noise Score",
                                    "Mean / Max Effective FFT Score",
                                    "ELA Used in Scoring",
                                    "Mean / Max Module 3 Score",
                                    "Overall Module 3 Suspicion",
                                    "Module 3 Manifest File",
                                ],
                                "Observed Value": [
                                    m3_res["forensics_metadata"]["case_id"],
                                    m3_res["forensics_metadata"]["timestamp"],
                                    ", ".join(m3_res["module3_settings"]["active_signals"]),
                                    f"{m3_sum['mean_noise_score']:.4f} / {m3_sum['max_noise_score']:.4f}",
                                    f"{m3_sum['mean_fft_score']:.4f} / {m3_sum['max_fft_score']:.4f}",
                                    "Yes (Native JPEG)" if m3_sum.get("ela_used_in_scoring") else "No (Non-JPEG / Video PNG)",
                                    f"{m3_sum['mean_module3_score']:.4f} / {m3_sum['max_module3_score']:.4f}",
                                    str(m3_sum["overall_suspicion"]).upper(),
                                    m3_res["artifacts_directory"]["manifest"],
                                ],
                            }
                        )
                with m3_c2:
                    m3_manifest_str = json.dumps(m3_res, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="⬇️ Download module3_manifest.json",
                        data=m3_manifest_str,
                        file_name=f"{m3_res['forensics_metadata']['case_id']}_module3_manifest.json",
                        mime="application/json",
                        type="primary",
                    )
else:
    st.info(
        "👈 Please configure settings in the left sidebar, click 'Apply Settings (Lock)', and upload evidence to begin analysis."
    )
