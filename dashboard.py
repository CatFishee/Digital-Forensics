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
from src.module_3 import ELAAnalyzer, FFTAnalyzer, Module3ForensicTranslator

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
    Execute Module 3 (ELA + 2D-FFT) on unedited raw frames (frame_path).
    - ELA uses Module 2 face_bboxes purely as an in-memory coordinate mask.
    - 2D-FFT runs on the full raw frame AND unwarped raw face bbox crops (no affine warping).
    - Deterministic rule-based plain-English interpretations are attached to each record.
    """
    manifest_dir = _get_manifest_dir(m2_res)
    ela_dir = manifest_dir / "ela"
    fft_dir = manifest_dir / "fft"
    fft_crops_dir = fft_dir / "raw_crops"
    ela_dir.mkdir(parents=True, exist_ok=True)
    fft_dir.mkdir(parents=True, exist_ok=True)

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

        # 1. ELA on raw frame with Module 2 face_bboxes mask
        ela_res = ela_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(ela_dir),
            face_bboxes=face_bboxes if face_bboxes else None,
            prefix=frame_stem,
        )

        # 2. Full-frame 2D-FFT on raw frame
        full_fft_res = fft_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(fft_dir),
            prefix=f"{frame_stem}_frame",
            save_radial_plot=True,
        )

        # 3. Unwarped raw face bbox 2D-FFT (direct pixel slice, zero affine interpolation)
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
        combined_m3_score = round(
            0.5 * ela_res["ela_score"] + 0.5 * effective_fft_score, 4
        )

        rec_name = f"{frame_stem}_module3_record.json"
        if rec_name in used_names:
            rec_name = f"{frame_stem}_{idx + 1:05d}_module3_record.json"
        used_names.add(rec_name)
        rec_path = manifest_dir / rec_name

        frame_m3_record = {
            "frame_index": idx + 1,
            "frame_path": frame_path,
            "ela": ela_res,
            "fft_full_frame": full_fft_res,
            "fft_unwarped_faces": face_raw_ffts,
            "summary": {
                "ela_score": ela_res["ela_score"],
                "ela_suspicion": ela_res["suspicion_level"],
                "frame_fft_score": full_fft_res["fft_score"],
                "max_face_fft_score": max_face_fft,
                "effective_fft_score": effective_fft_score,
                "fft_suspicion": FFTAnalyzer._compute_suspicion_level(effective_fft_score),
                "combined_module3_score": combined_m3_score,
                "combined_suspicion": ELAAnalyzer._compute_suspicion_level(combined_m3_score),
            },
            "record_json_path": str(rec_path.resolve()),
        }
        frame_m3_record["interpretation"] = translator.interpret_frame(frame_m3_record)

        rec_path.write_text(
            json.dumps(frame_m3_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        frame_results.append(frame_m3_record)

    ela_scores = [r["summary"]["ela_score"] for r in frame_results]
    fft_scores = [r["summary"]["effective_fft_score"] for r in frame_results]
    combined_scores = [r["summary"]["combined_module3_score"] for r in frame_results]

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
            "ela_quality": ela_quality,
            "ela_face_bbox_mask": True,
            "fft_modes": ["Full Raw Frame", "Unwarped Raw Face BBox Crop"],
        },
        "execution_summary": {
            "frames_analyzed": len(frame_results),
            "mean_ela_score": mean_ela,
            "max_ela_score": max_ela,
            "mean_fft_score": mean_fft,
            "max_fft_score": max_fft,
            "mean_module3_score": mean_comb,
            "max_module3_score": max_comb,
            "overall_suspicion": ELAAnalyzer._compute_suspicion_level(max_comb),
        },
        "artifacts_directory": {
            "root": str(manifest_dir.resolve()),
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


def _render_translation_block(
    translation: dict | None,
    title: str = "Plain-English Forensic Interpretation",
) -> None:
    """Render deterministic rule-based forensic explanations for non-expert users."""
    if not translation:
        return

    st.markdown(f"###### 💡 {title}")
    sev = str(translation.get("severity", "low")).lower()
    headline = translation.get("headline", "")
    if sev == "high":
        st.error(f"**Summary:** {headline}")
    elif sev == "medium":
        st.warning(f"**Summary:** {headline}")
    else:
        st.success(f"**Summary:** {headline}")

    for item in translation.get("findings", []):
        item_sev = item.get("severity", "info")
        text = f"**{item.get('title', 'Finding')}:** {item.get('detail', '')}"
        if item_sev == "alert":
            st.error(text)
        elif item_sev == "warning":
            st.warning(text)
        elif item_sev == "ok":
            st.success(text)
        else:
            st.info(text)

    guidance = translation.get("user_guidance")
    if guidance:
        st.caption(f"ℹ️ {guidance}")


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


def _render_module3_ela(ela_res: dict, ela_translation: dict | None = None) -> None:
    """Render parsed Module 3.1 ELA analysis, images, and deterministic findings."""
    st.markdown("##### 🔥 Error Level Analysis (ELA) on Raw Frame")
    if ela_res.get("is_jpeg"):
        st.success(ela_res.get("note", ""))
    else:
        st.warning(ela_res.get("note", ""))

    c1, c2 = st.columns(2)
    with c1:
        raw_p = ela_res.get("ela_raw_path")
        if raw_p and os.path.exists(raw_p):
            st.image(raw_p, caption="Amplified ELA Residual (RGB)", use_container_width=True)
    with c2:
        heat_p = ela_res.get("ela_heatmap_path")
        if heat_p and os.path.exists(heat_p):
            st.image(heat_p, caption="ELA Error Heatmap (Inferno)", use_container_width=True)

    _render_translation_block(
        ela_translation,
        title="Plain-English ELA Interpretation",
    )

    metrics = ela_res.get("metrics", {})
    face_mean = metrics.get("face_region_mean")
    bg_mean = metrics.get("bg_region_mean")
    contrast = metrics.get("face_bg_contrast")

    st.markdown("###### 📊 Quantitative ELA Metrics")
    st.table(
        {
            "ELA Metric": [
                "ELA Suspicion Score",
                "Suspicion Level",
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
                f"{ela_res['ela_score']:.4f}",
                str(ela_res["suspicion_level"]).upper(),
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
    """Render parsed Module 3.2 2D-FFT (Full Frame & Unwarped Raw Face BBox Crops)."""
    full_fft = frame_m3["fft_full_frame"]
    face_ffts = frame_m3.get("fft_unwarped_faces", [])
    fft_translation = frame_m3.get("interpretation", {}).get("fft")

    _render_translation_block(
        fft_translation,
        title="Plain-English 2D-FFT Interpretation",
    )

    st.markdown("##### 📡 Full Raw Frame 2D-FFT Spectrum")
    c1, c2 = st.columns(2)
    with c1:
        spec_p = full_fft.get("fft_spectrum_path")
        if spec_p and os.path.exists(spec_p):
            st.image(spec_p, caption="2D-FFT Log Magnitude Spectrum (Full Frame)", use_container_width=True)
    with c2:
        rad_p = full_fft.get("fft_radial_plot_path")
        if rad_p and os.path.exists(rad_p):
            st.image(rad_p, caption="1D Azimuthal Power Profile (Full Frame)", use_container_width=True)

    f_metrics = full_fft.get("metrics", {})
    st.table(
        {
            "Full-Frame FFT Metric": [
                "Full-Frame FFT Score",
                "Suspicion Level",
                "Frame Resolution",
                "High-Frequency Energy Ratio",
                "Max Periodic Spike Z-Score",
                "Top-20 Spike Mean Z-Score",
                "Spike Density (>4σ)",
                "Radial Curve Roughness",
            ],
            "Observed Value": [
                f"{full_fft['fft_score']:.4f}",
                str(full_fft["suspicion_level"]).upper(),
                f"{full_fft['image_resolution'][0]} x {full_fft['image_resolution'][1]} px",
                str(f_metrics.get("hf_energy_ratio")),
                str(f_metrics.get("max_spike_z")),
                str(f_metrics.get("top20_spike_mean_z")),
                str(f_metrics.get("spike_density")),
                str(f_metrics.get("radial_roughness")),
            ],
        }
    )

    st.markdown("##### 🎯 Unwarped Raw Face BBox 2D-FFT (Zero Affine / Bicubic Distortion)")
    if not face_ffts:
        st.info(
            "No valid face bounding boxes (>= 32x32 px) in this frame; full-frame 2D-FFT is used."
        )
        return

    for f_fft in face_ffts:
        with st.container(border=True):
            st.markdown(
                f"**{f_fft['global_face_id']}** — Unwarped BBox `{f_fft['bbox']}` | "
                f"FFT Score: `{f_fft['fft_score']:.4f}` (`{str(f_fft['suspicion_level']).upper()}`)"
            )
            col_crop, col_spec, col_rad = st.columns([0.8, 1.1, 1.1])
            with col_crop:
                if os.path.exists(f_fft["raw_crop_path"]):
                    st.image(
                        f_fft["raw_crop_path"],
                        caption=f"Unwarped Raw Slice ({f_fft['image_resolution'][0]}x{f_fft['image_resolution'][1]})",
                        use_container_width=True,
                    )
            with col_spec:
                if f_fft.get("fft_spectrum_path") and os.path.exists(f_fft["fft_spectrum_path"]):
                    st.image(
                        f_fft["fft_spectrum_path"],
                        caption="Face ROI 2D-FFT Spectrum",
                        use_container_width=True,
                    )
            with col_rad:
                if f_fft.get("fft_radial_plot_path") and os.path.exists(f_fft["fft_radial_plot_path"]):
                    st.image(
                        f_fft["fft_radial_plot_path"],
                        caption="Face ROI 1D Radial Spectrum",
                        use_container_width=True,
                    )

            fm = f_fft.get("metrics", {})
            st.table(
                {
                    "Face-ROI FFT Metric": [
                        "High-Frequency Energy Ratio",
                        "Max Periodic Spike Z-Score",
                        "Top-20 Spike Mean Z-Score",
                        "Spike Density (>4σ)",
                        "Radial Curve Roughness",
                    ],
                    "Observed Value": [
                        str(fm.get("hf_energy_ratio")),
                        str(fm.get("max_spike_z")),
                        str(fm.get("top20_spike_mean_z")),
                        str(fm.get("spike_density")),
                        str(fm.get("radial_roughness")),
                    ],
                }
            )


def _render_module3_frame_record(frame_m3: dict) -> None:
    """Render parsed Module 3 frame-level summary record and plain-English takeaways without raw JSON."""
    st.markdown("##### 📄 Frame-Level Module 3 Forensic Record")

    interp = frame_m3.get("interpretation", {})
    if interp:
        sev = str(interp.get("severity", "low")).lower()
        headline = interp.get("headline", "")
        if sev == "high":
            st.error(f"**Frame Conclusion:** {headline}")
        elif sev == "medium":
            st.warning(f"**Frame Conclusion:** {headline}")
        else:
            st.success(f"**Frame Conclusion:** {headline}")

        for takeaway in interp.get("key_takeaways", []):
            st.info(takeaway)

    s = frame_m3["summary"]
    st.table(
        {
            "Module 3 Property": [
                "Frame Index",
                "Source Raw Frame",
                "ELA Score (Suspicion)",
                "Full-Frame FFT Score",
                "Max Unwarped Face FFT Score",
                "Effective FFT Score (Suspicion)",
                "Combined Module 3 Score (Suspicion)",
                "Saved Module 3 Record (.json)",
            ],
            "Observed Value": [
                str(frame_m3["frame_index"]),
                frame_m3["frame_path"],
                f"{s['ela_score']:.4f} ({str(s['ela_suspicion']).upper()})",
                f"{s['frame_fft_score']:.4f}",
                f"{s['max_face_fft_score']:.4f}" if s["max_face_fft_score"] is not None else "N/A",
                f"{s['effective_fft_score']:.4f} ({str(s['fft_suspicion']).upper()})",
                f"{s['combined_module3_score']:.4f} ({str(s['combined_suspicion']).upper()})",
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
    "Module 3 ELA JPEG Quality",
    min_value=50,
    max_value=99,
    value=int(current_cfg.get("ela_quality", 90)),
    step=1,
    disabled=is_locked,
    help="Re-compression JPEG quality used by Error Level Analysis (ELA) on raw frames.",
)

st.sidebar.markdown("---")
# Lock / Unlock button to control forensic pipeline execution
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
        # Clear cached results so the user can edit parameters
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

    # Save a single timestamped copy into data/raw/ per uploaded file (avoid duplicates on reruns)
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
            f"{active_cfg['output_size']}_{active_cfg['max_frames']}_{active_cfg['ela_quality']}"
        )

        # Execute Module 2 & Module 3 only when locked and not yet cached
        if st.session_state.get("m2_cache_key") != cache_key:
            with st.spinner(
                f"Executing Module 2 ({active_cfg['detector_backend'].upper()}) & Module 3 (ELA + 2D-FFT on raw frames)..."
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
            # MODULE 3: PIXEL-LEVEL FORENSICS (ELA & 2D-FFT)
            # =================================================
            if m3_res:
                st.divider()
                st.header("🟠 Module 3: Pixel-Level Forensics (ELA & 2D-FFT)")
                st.caption(
                    "Executed on unedited raw frames (`frame_path`). ELA uses Module 2 `face_bboxes` as a coordinate mask; "
                    "2D-FFT analyzes both the full raw frame and unwarped raw face slices."
                )

                m3_sum = m3_res["execution_summary"]
                mk1, mk2, mk3, mk4, mk5 = st.columns(5)
                mk1.metric("Frames Analyzed", m3_sum["frames_analyzed"])
                mk2.metric("Mean / Max ELA", f"{m3_sum['mean_ela_score']:.3f} / {m3_sum['max_ela_score']:.3f}")
                mk3.metric("Mean / Max FFT", f"{m3_sum['mean_fft_score']:.3f} / {m3_sum['max_fft_score']:.3f}")
                mk4.metric("Max Combined Score", f"{m3_sum['max_module3_score']:.3f}")
                mk5.metric("Module 3 Suspicion", str(m3_sum["overall_suspicion"]).upper())

                _render_translation_block(
                    m3_res.get("interpretation"),
                    title="Overall Module 3 Plain-English Takeaways",
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
                        "🔥 1. Error Level Analysis (ELA)",
                        "📡 2. 2D-FFT Spectrum (Frame & Raw Face)",
                        "📄 3. Frame Module 3 Record",
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
                    st.write("")
                    st.info(
                        f"**Active Frame:** #{m3_frame_idx} (`{Path(current_m3['frame_path']).name}`) | "
                        f"**ELA:** `{cur_s['ela_score']:.4f}` (`{cur_s['ela_suspicion'].upper()}`) | "
                        f"**Effective FFT:** `{cur_s['effective_fft_score']:.4f}` (`{cur_s['fft_suspicion'].upper()}`)"
                    )

                    if m3_mode == m3_modes[0]:
                        _render_module3_ela(
                            current_m3["ela"],
                            ela_translation=current_m3.get("interpretation", {}).get("ela"),
                        )
                    elif m3_mode == m3_modes[1]:
                        _render_module3_fft(current_m3)
                    elif m3_mode == m3_modes[2]:
                        _render_module3_frame_record(current_m3)

                st.divider()
                st.subheader("📋 Module 3 Forensic Audit Summary")
                m3_c1, m3_c2 = st.columns([2, 1])
                with m3_c1:
                    st.table(
                        {
                            "Module 3 Property": [
                                "Case Identifier",
                                "Audit Timestamp",
                                "Input Source",
                                "ELA JPEG Quality",
                                "Mean / Max ELA Score",
                                "Mean / Max Effective FFT Score",
                                "Mean / Max Module 3 Score",
                                "Overall Module 3 Suspicion",
                                "Module 3 Manifest File",
                            ],
                            "Observed Value": [
                                m3_res["forensics_metadata"]["case_id"],
                                m3_res["forensics_metadata"]["timestamp"],
                                m3_res["module3_settings"]["input_source"],
                                f"Q={m3_res['module3_settings']['ela_quality']}",
                                f"{m3_sum['mean_ela_score']:.4f} / {m3_sum['max_ela_score']:.4f}",
                                f"{m3_sum['mean_fft_score']:.4f} / {m3_sum['max_fft_score']:.4f}",
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
