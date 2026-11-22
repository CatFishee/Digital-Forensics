from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
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
    BoundaryAnalyzer,
    FFTAnalyzer,
    Module3ForensicTranslator,
    NoiseAnalyzer,
    TemporalAnalyzer,
)

TEMP_DIR = Path("data/raw")
PREVIEW_DIR = TEMP_DIR / "_previews"
PROCESSED_ROOT = Path("data/processed")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
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


def _get_browser_playable_video(src_path: str, file_sha: str) -> tuple[str, bool]:
    """
    Ensure HTML5 browser playback for dataset videos (e.g. mp4v, HEVC, AVI, MKV)
    by creating a cached display-only H.264/yuv420p preview without modifying
    the original evidence file used for hashes and metadata.
    """
    p = Path(src_path)
    if p.suffix.lower() == ".mp4" and shutil.which("ffprobe"):
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,pix_fmt",
                    "-of",
                    "json",
                    src_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0 and probe.stdout:
                streams = json.loads(probe.stdout).get("streams", [])
                if streams:
                    codec = str(streams[0].get("codec_name", "")).lower()
                    pix_fmt = str(streams[0].get("pix_fmt", "")).lower()
                    if codec == "h264" and pix_fmt == "yuv420p":
                        return src_path, False
        except Exception:
            pass

    preview_path = PREVIEW_DIR / f"{file_sha[:12]}_preview.mp4"
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return str(preview_path), True

    if not shutil.which("ffmpeg"):
        return src_path, False

    base_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    try:
        res = subprocess.run(base_cmd, capture_output=True, timeout=60)
        if res.returncode == 0 and preview_path.exists() and preview_path.stat().st_size > 0:
            return str(preview_path), True

        no_audio_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview_path),
        ]
        res2 = subprocess.run(no_audio_cmd, capture_output=True, timeout=60)
        if res2.returncode == 0 and preview_path.exists() and preview_path.stat().st_size > 0:
            return str(preview_path), True
    except Exception:
        pass

    return src_path, False


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


def _run_module3_on_raw_frames(m2_res: dict) -> dict:
    """
    Execute Module 3 on unedited raw frames & consecutive bursts:
      1. BoundaryAnalyzer (Face-Boundary Blending & Chrominance Mismatch)
      2. TemporalAnalyzer (Multi-Face Video Burst Flicker & Landmark Jitter)
      3. NoiseAnalyzer (SRM Pixel Grain Consistency)
      4. FFTAnalyzer (2D-FFT & 1/f Spectral Decay — Unwarped Face Crops + Full Frame)
    """
    manifest_dir = _get_manifest_dir(m2_res)
    boundary_dir = manifest_dir / "boundary"
    temporal_dir = manifest_dir / "temporal"
    noise_dir = manifest_dir / "noise"
    fft_dir = manifest_dir / "fft"
    fft_crops_dir = fft_dir / "raw_crops"
    for d in (boundary_dir, temporal_dir, noise_dir, fft_dir):
        d.mkdir(parents=True, exist_ok=True)

    boundary_analyzer = BoundaryAnalyzer()
    temporal_analyzer = TemporalAnalyzer()
    noise_analyzer = NoiseAnalyzer()
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
        burst_frames = frame_rec.get("burst_frames") or [
            {"burst_index": 0, "frame_path": frame_path, "faces": faces}
        ]

        # 1. Face-Boundary Blending & Chrominance Mismatch
        boundary_res = boundary_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(boundary_dir),
            faces=faces,
            prefix=frame_stem,
        )

        # 2. Multi-Face Temporal Face Flicker & Inter-Frame Consistency
        temporal_res = temporal_analyzer.analyze(
            burst_frames=burst_frames,
            output_dir=str(temporal_dir),
            prefix=frame_stem,
        )

        # 3. SRM Noise Residual Analysis
        noise_res = noise_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(noise_dir),
            face_bboxes=face_bboxes if face_bboxes else None,
            prefix=frame_stem,
        )

        # 4. Full-frame 2D-FFT
        full_fft_res = fft_analyzer.analyze(
            image_path=frame_path,
            output_dir=str(fft_dir),
            prefix=f"{frame_stem}_frame",
            save_radial_plot=True,
            trim_borders=True,
        )

        # 5. Unwarped raw face bbox 2D-FFT
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

        active_scores = [noise_res["noise_score"], effective_fft_score]
        boundary_eval = (boundary_res.get("metrics", {}).get("faces_evaluated") or 0) > 0
        if boundary_eval:
            active_scores.append(boundary_res["boundary_score"])

        temporal_eval = bool(temporal_res.get("applicable", False)) and (
            (temporal_res.get("metrics", {}).get("tracked_frames") or 0) >= 2
        )
        if temporal_eval:
            active_scores.append(temporal_res["temporal_score"])

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
            "boundary": boundary_res,
            "temporal": temporal_res,
            "noise": noise_res,
            "fft_full_frame": full_fft_res,
            "fft_unwarped_faces": face_raw_ffts,
            "summary": {
                "boundary_score": boundary_res["boundary_score"],
                "boundary_suspicion": boundary_res["suspicion_level"],
                "boundary_evaluated": boundary_eval,
                "temporal_score": temporal_res["temporal_score"],
                "temporal_suspicion": temporal_res["suspicion_level"],
                "temporal_applicable": bool(temporal_res.get("applicable", False)),
                "noise_score": noise_res["noise_score"],
                "noise_suspicion": noise_res["suspicion_level"],
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

    boundary_scores = [r["summary"]["boundary_score"] for r in frame_results]
    temporal_scores = [
        r["summary"]["temporal_score"]
        for r in frame_results
        if r["summary"]["temporal_applicable"]
    ]
    noise_scores = [r["summary"]["noise_score"] for r in frame_results]
    fft_scores = [r["summary"]["effective_fft_score"] for r in frame_results]
    combined_scores = [r["summary"]["combined_module3_score"] for r in frame_results]

    mean_bound = round(float(np.mean(boundary_scores)), 4) if boundary_scores else 0.0
    max_bound = round(float(np.max(boundary_scores)), 4) if boundary_scores else 0.0
    mean_temp = round(float(np.mean(temporal_scores)), 4) if temporal_scores else 0.0
    max_temp = round(float(np.max(temporal_scores)), 4) if temporal_scores else 0.0
    mean_noise = round(float(np.mean(noise_scores)), 4) if noise_scores else 0.0
    max_noise = round(float(np.max(noise_scores)), 4) if noise_scores else 0.0
    mean_fft = round(float(np.mean(fft_scores)), 4) if fft_scores else 0.0
    max_fft = round(float(np.max(fft_scores)), 4) if fft_scores else 0.0
    mean_comb = round(float(np.mean(combined_scores)), 4) if combined_scores else 0.0
    max_comb = round(float(np.max(combined_scores)), 4) if combined_scores else 0.0
    temporal_analyzed = len(temporal_scores) > 0

    active_signals = [
        "Face-Boundary Blending & Color Match",
        "SRM Pixel Grain (Noise)",
        "2D-FFT Spectrum (1/f + Chroma)",
    ]
    if temporal_analyzed:
        active_signals.insert(1, "Temporal Face Flicker (Video Bursts)")

    m3_manifest_path = manifest_dir / "module3_manifest.json"
    m3_manifest = {
        "forensics_metadata": {
            "case_id": m2_res["forensics_metadata"]["case_id"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "media_type": m2_res["forensics_metadata"]["media_type"],
        },
        "module3_settings": {
            "input_source": "Unedited Raw Frames & Consecutive Bursts",
            "active_signals": active_signals,
            "fft_modes": ["Unwarped Raw Face BBox Crop", "Full Raw Frame"],
        },
        "execution_summary": {
            "frames_analyzed": len(frame_results),
            "mean_boundary_score": mean_bound,
            "max_boundary_score": max_bound,
            "mean_temporal_score": mean_temp,
            "max_temporal_score": max_temp,
            "temporal_analyzed": temporal_analyzed,
            "mean_noise_score": mean_noise,
            "max_noise_score": max_noise,
            "mean_fft_score": mean_fft,
            "max_fft_score": max_fft,
            "mean_module3_score": mean_comb,
            "max_module3_score": max_comb,
            "overall_suspicion": NoiseAnalyzer._compute_suspicion_level(max_comb),
        },
        "artifacts_directory": {
            "root": str(manifest_dir.resolve()),
            "boundary": str(boundary_dir.resolve()),
            "temporal": str(temporal_dir.resolve()),
            "noise": str(noise_dir.resolve()),
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


def _render_m3_kpi(
    col, label: str, mean_score: float, max_score: float, is_multi: bool
) -> None:
    """Render a clear Module 3 KPI metric (Peak% + Avg/Peak subtitle for multi-frame videos)."""
    peak_pct = f"{max_score * 100:.0f}%"
    if is_multi:
        avg_pct = f"{mean_score * 100:.0f}%"
        col.metric(
            label,
            peak_pct,
            delta=f"Avg {avg_pct} • Peak {peak_pct}",
            delta_color="off",
        )
    else:
        col.metric(label, peak_pct)


def _make_pill_html(
    label: str,
    active: bool,
    sev: str,
    score: float,
    na_text: str = "N/A",
    frame_idx: int | None = None,
) -> str:
    """Build an HTML colored pill badge matching the Module 3 quick-glance palette."""
    if not active:
        return f'<span class="m3-pill pill-na">⚪ {label}: {na_text}</span>'
    sev_l = str(sev).lower()
    pct = f"{float(score) * 100:.0f}%"
    frame_part = f"Frame #{frame_idx} — " if frame_idx is not None else ""
    if sev_l == "high":
        return f'<span class="m3-pill pill-high">🔴 {label}: {frame_part}HIGH ({pct})</span>'
    if sev_l == "medium":
        return f'<span class="m3-pill pill-med">🟡 {label}: {frame_part}MOD ({pct})</span>'
    return f'<span class="m3-pill pill-low">🟢 {label}: {frame_part}LOW ({pct})</span>'


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


def _render_frame_status_box(frame_idx: int, frame_m3: dict) -> None:
    """
    Render the active Frame status box with color-coded pill badges matching tabs 1..4,
    so high/moderate suspicion tabs stand out at a quick glance.
    """
    s = frame_m3["summary"]
    label, icon, css_cls = _suspicion_display(
        s["combined_suspicion"], s["combined_module3_score"]
    )
    file_name = Path(frame_m3["frame_path"]).name

    pills = [
        _make_pill_html(
            "1. Face Border",
            bool(s.get("boundary_evaluated")),
            s["boundary_suspicion"],
            s["boundary_score"],
            "No Face",
        ),
        _make_pill_html(
            "2. Video Flicker",
            bool(s.get("temporal_applicable")),
            s["temporal_suspicion"],
            s["temporal_score"],
            "Image Only",
        ),
        _make_pill_html(
            "3. Pixel Grain",
            True,
            s["noise_suspicion"],
            s["noise_score"],
        ),
        _make_pill_html(
            "4. Frequency",
            True,
            s["fft_suspicion"],
            s["effective_fft_score"],
        ),
    ]

    st.markdown(
        f"""
        <div class="status-banner {css_cls}">
            <div class="status-title">{icon} Frame #{frame_idx} Status: <span>{label}</span>
                <span class="status-file">({file_name})</span>
            </div>
            <div class="pill-row">{''.join(pills)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_and_details(
    translation: dict | None,
    verdict_title: str = "Verdict",
    expander_label: str = "📖 Click to see detailed explanation",
) -> None:
    """
    Show a short, colored verdict box first, followed by a collapsed expander
    containing simple plain-English details and optional peak-frame pill badges.
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
    peak_frames = translation.get("peak_frames", []) or []
    multi_frame = bool(translation.get("multi_frame", False))
    guidance = translation.get("user_guidance")
    if not findings and not peak_frames and not guidance:
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

        if peak_frames:
            heading = "Most Notable Frame(s):" if multi_frame else "Signal Breakdown:"
            st.markdown(f"📌 **{heading}**")
            pills_html = [
                _make_pill_html(
                    pf["label"],
                    bool(pf.get("applicable", True)),
                    pf["severity"],
                    pf["score"],
                    na_text="No Face",
                    frame_idx=pf["frame_index"] if multi_frame else None,
                )
                for pf in peak_frames
            ]
            st.markdown(
                f'<div class="pill-row">{"".join(pills_html)}</div>',
                unsafe_allow_html=True,
            )

        if guidance:
            st.markdown("---")
            st.caption(f"💡 {guidance}")


def _render_frame_forensic_record(
    current_frame: dict, record_file_path: str | None = None
) -> None:
    """Parse and render a frame's Module 2 forensic record neatly without raw JSON."""
    st.markdown("##### 📄 Frame Details")

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
    m1.metric("Image File", frame_name)
    m2.metric("Image Size", res_str)
    m3.metric("Faces Found", faces_detected)
    m4.metric("Face Finder", primary_detector)

    with st.expander("🗂️ Technical File Paths & Face Coordinates", expanded=False):
        st.table(
            {
                "Property": [
                    "Source Frame File",
                    "Source Frame Path",
                    "Annotated Overlay Path",
                    "Burst Frames Recorded",
                    "Saved Record (.json)",
                ],
                "Value": [
                    frame_name,
                    frame_path or "N/A",
                    current_frame.get("annotated_frame_path") or "N/A",
                    str(len(current_frame.get("burst_frames", []))),
                    record_file_path or "N/A",
                ],
            }
        )

        if faces:
            summary_rows = {
                "Face ID": [],
                "Detector": [],
                "Confidence": [],
                "Box [X1, Y1, X2, Y2]": [],
                "Cropped Size": [],
            }
            for f_info in faces:
                bbox = f_info.get("bbox", [0, 0, 0, 0])
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
                summary_rows["Box [X1, Y1, X2, Y2]"].append(
                    f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if len(bbox) >= 4 else "N/A"
                )
                summary_rows["Cropped Size"].append(aligned_str)
            st.table(summary_rows)


def _render_module3_boundary(
    boundary_res: dict, boundary_translation: dict | None = None
) -> None:
    """Render Face-Boundary Blending & Color Match: Visuals -> Verdict -> Details -> Tucked Metrics."""
    st.markdown("##### 🎭 Face-Border Blending & Skin Color Check")
    st.caption(
        "Checks along the lower face outline (cheeks and jawline) for pasted blending halos "
        "or skin-tone mismatches against the neck and surroundings."
    )

    ov_p = boundary_res.get("boundary_overlay_path")
    hm_p = boundary_res.get("boundary_heatmap_path")
    if ov_p or hm_p:
        c1, c2 = st.columns(2)
        with c1:
            if ov_p and os.path.exists(ov_p):
                st.image(
                    ov_p,
                    caption="Inspected Face Outline Zone (Cheeks & Jawline)",
                    use_container_width=True,
                )
        with c2:
            if hm_p and os.path.exists(hm_p):
                st.image(
                    hm_p,
                    caption="Border & Color-Shift Heatmap (Brighter = Stronger Transition)",
                    use_container_width=True,
                )

    st.write("")
    _render_verdict_and_details(
        boundary_translation,
        verdict_title="Face Border & Color Verdict",
        expander_label="📖 Why did it say this? (Simple explanation)",
    )

    m = boundary_res.get("metrics", {}) or {}
    with st.expander("📊 Technical Face-Boundary Metrics (For Experts)", expanded=False):
        st.table(
            {
                "Boundary Metric": [
                    "Boundary Suspicion Score",
                    "Faces Evaluated",
                    "Pose / Framing Warning",
                    "Seam Halo Ratio",
                    "Skin-vs-Context Color Diff (Lab)",
                    "Seam Chrominance Bleed",
                    "Inner-vs-Outer Sharpness Ratio",
                ],
                "Observed Value": [
                    f"{boundary_res['boundary_score']:.4f} ({boundary_res['boundary_score'] * 100:.0f}%)",
                    str(m.get("faces_evaluated", 0)),
                    "Yes (Extreme angle or covered face)" if m.get("unreliable_face_warning") else "No",
                    str(m.get("seam_halo_ratio")) if m.get("seam_halo_ratio") is not None else "N/A",
                    str(m.get("skin_context_chroma_diff")) if m.get("skin_context_chroma_diff") is not None else "N/A",
                    str(m.get("seam_chroma_bleed")) if m.get("seam_chroma_bleed") is not None else "N/A",
                    str(m.get("inner_outer_sharpness_ratio")) if m.get("inner_outer_sharpness_ratio") is not None else "N/A",
                ],
            }
        )


def _render_module3_temporal(
    temporal_res: dict, temporal_translation: dict | None = None
) -> None:
    """Render Temporal Face Flicker: Side-by-side card for every tracked face + Verdict + Details + Metrics."""
    st.markdown("##### 🎞️ Frame-to-Frame Face Flicker Check (Video Only)")
    st.caption(
        "Follows each face across consecutive frames around this frame to spot unstable features, "
        "skin-texture flickering, or sudden brightness/color popping."
    )

    tracked_faces = temporal_res.get("faces", []) or []
    untracked_faces = temporal_res.get("untracked_faces", []) or []

    if temporal_res.get("applicable", False):
        if not tracked_faces:
            st.info("No face could be tracked across at least 2 consecutive frames in this burst.")
        else:
            for face_item in tracked_faces:
                f_gid = face_item.get("global_face_id") or f"Face #{face_item.get('face_index', 0)}"
                f_score = float(face_item.get("temporal_score", 0.0))
                f_sev = face_item.get("suspicion_level", "low")
                f_label, f_icon, _ = _suspicion_display(f_sev, f_score)
                f_tracked_cnt = face_item.get("tracked_frames", 0)
                f_total_cnt = face_item.get("total_burst_frames", 0)
                unreliable = bool(face_item.get("unreliable", False))

                with st.container(border=True):
                    header_txt = (
                        f"**{f_gid}** — Video Flicker: {f_icon} `{f_label}` "
                        f"({f_tracked_cnt}/{f_total_cnt} frames tracked)"
                    )
                    if unreliable:
                        header_txt += " — *⚠️ Extreme angle or covered face*"
                    st.markdown(header_txt)

                    col_crop, col_heat, col_plot = st.columns([0.8, 1.1, 1.3])
                    with col_crop:
                        rc_path = face_item.get("raw_crop_path")
                        if rc_path and os.path.exists(rc_path):
                            st.image(
                                rc_path,
                                caption=f"Raw Face Crop ({f_gid})",
                                use_container_width=True,
                            )
                        else:
                            st.caption(f"Raw Crop ({f_gid}) N/A")
                    with col_heat:
                        hm_path = face_item.get("temporal_heatmap_path")
                        if hm_path and os.path.exists(hm_path):
                            st.image(
                                hm_path,
                                caption="Skin Texture Flicker Map",
                                use_container_width=True,
                            )
                        else:
                            st.caption("Flicker Heatmap N/A")
                    with col_plot:
                        pl_path = face_item.get("temporal_plot_path")
                        if pl_path and os.path.exists(pl_path):
                            st.image(
                                pl_path,
                                caption="Color, Brightness & Texture Stability",
                                use_container_width=True,
                            )
                        else:
                            st.caption("Stability Plot N/A")

        if untracked_faces:
            untracked_desc = [
                f"**{uf.get('global_face_id', 'Face')}**: {uf.get('reason', 'Dropped')}"
                for uf in untracked_faces
            ]
            st.info("ℹ️ " + " | ".join(untracked_desc))

    st.write("")
    _render_verdict_and_details(
        temporal_translation,
        verdict_title="Video Flicker Verdict",
        expander_label="📖 Why did it say this? (Simple explanation)",
    )

    m = temporal_res.get("metrics", {}) or {}
    faces_tracked_cnt = int(m.get("faces_tracked_count", 0) or 0)
    peak_gid = m.get("peak_face_id") or (f"Face #{m.get('peak_face_index')}" if m.get("peak_face_index") is not None else "N/A")
    tracked_str = f"{faces_tracked_cnt} (Peak: {peak_gid})" if faces_tracked_cnt > 0 else "0"

    with st.expander("📊 Technical Temporal Metrics (For Experts)", expanded=False):
        st.table(
            {
                "Temporal Metric": [
                    "Applicable",
                    "Burst Frames / Peak Face Steps",
                    "Faces Tracked Independently",
                    "Pose / Framing Warning",
                    "Peak Temporal Suspicion Score",
                    "2nd-Order Landmark Jitter (Peak Face)",
                    "2nd-Order BBox Jitter (Peak Face)",
                    "Motion-Aligned Texture Flicker (Peak Face)",
                    "2nd-Order Color Popping Lab (Peak Face)",
                    "2nd-Order Brightness Popping (Peak Face)",
                ],
                "Observed Value": [
                    "Yes (Video Burst)" if temporal_res.get("applicable") else "No (Single Image)",
                    f"{temporal_res.get('burst_frames_count', 0)} / {m.get('tracked_frames', 0)}",
                    tracked_str,
                    "Yes (Extreme angle or covered face)" if m.get("unreliable_face_warning") else "No",
                    f"{temporal_res['temporal_score']:.4f} ({temporal_res['temporal_score'] * 100:.0f}%)",
                    str(m.get("landmark_jitter")) if m.get("landmark_jitter") is not None else "N/A",
                    str(m.get("bbox_jitter")) if m.get("bbox_jitter") is not None else "N/A",
                    str(m.get("texture_flicker_ratio")) if m.get("texture_flicker_ratio") is not None else "N/A",
                    str(m.get("color_flicker_lab")) if m.get("color_flicker_lab") is not None else "N/A",
                    str(m.get("brightness_flicker")) if m.get("brightness_flicker") is not None else "N/A",
                ],
            }
        )
        if len(tracked_faces) > 1:
            st.caption(
                "Per-face breakdown: "
                + ", ".join(
                    f"{tf['global_face_id']}: {tf['temporal_score'] * 100:.0f}% "
                    f"({tf['tracked_frames']}/{tf['total_burst_frames']} frames"
                    + (", extreme angle/border)" if tf.get("unreliable") else ")")
                    for tf in tracked_faces
                )
            )


def _render_module3_noise(noise_res: dict, noise_translation: dict | None = None) -> None:
    """Render Noise Consistency: Visuals -> Verdict -> Details -> Tucked Metrics."""
    st.markdown("##### 🔬 Pixel Grain Consistency Check")
    st.caption(
        "Hides colors and lighting to expose tiny camera grain differences—for example, "
        "an overly smooth AI face placed onto a natural grainy scene."
    )

    c1, c2 = st.columns(2)
    with c1:
        res_p = noise_res.get("noise_residual_path")
        if res_p and os.path.exists(res_p):
            st.image(res_p, caption="Isolated Pixel Grain (SRM Residual)", use_container_width=True)
    with c2:
        heat_p = noise_res.get("noise_heatmap_path")
        if heat_p and os.path.exists(heat_p):
            st.image(
                heat_p,
                caption="Grain Strength Map (Blue = Smooth, Yellow/Red = Heavy Grain)",
                use_container_width=True,
            )

    st.write("")
    _render_verdict_and_details(
        noise_translation,
        verdict_title="Pixel Grain Verdict",
        expander_label="📖 Why did it say this? (Simple explanation)",
    )

    m = noise_res.get("metrics", {}) or {}
    with st.expander("📊 Technical Pixel-Grain Metrics (For Experts)", expanded=False):
        st.table(
            {
                "Noise Metric": [
                    "Noise Suspicion Score",
                    "Noise P95 / P50 Ratio",
                    "Block Noise CV",
                    "Global Noise Median",
                    "Face Skin Noise (35th Pct)",
                    "Background Noise Median",
                    "Face-vs-Background Contrast",
                    "Face Smoother Than Background",
                    "Face / BG Chrominance Noise",
                    "Face-vs-BG Chroma Contrast",
                ],
                "Observed Value": [
                    f"{noise_res['noise_score']:.4f} ({noise_res['noise_score'] * 100:.0f}%)",
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
                ],
            }
        )


def _render_module3_fft(frame_m3: dict) -> None:
    """Render 2D-FFT: Unwarped Face Crops First -> Full Frame -> Verdict -> Details -> Tucked Metrics."""
    full_fft = frame_m3["fft_full_frame"]
    face_ffts = frame_m3.get("fft_unwarped_faces", [])
    fft_translation = frame_m3.get("interpretation", {}).get("fft")

    st.markdown("##### 🎯 Face Frequency Scan (Primary Check)")
    st.caption(
        "Scans each face directly from the raw image (without stretching) to spot repeating "
        "AI grid patterns or unnatural fine-detail curves."
    )

    if not face_ffts:
        st.info("No faces large enough (>= 32x32 px) were detected here; see the full-frame scan below.")
    else:
        for f_fft in face_ffts:
            label, icon, _ = _suspicion_display(f_fft["suspicion_level"], f_fft["fft_score"])
            with st.container(border=True):
                st.markdown(f"**{f_fft['global_face_id']}** — Frequency Suspicion: {icon} `{label}`")
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
                            caption="Face Detail Curve vs. Normal Camera Trend",
                            use_container_width=True,
                        )

    st.markdown("##### 📡 Full-Frame Frequency Scan")
    c1, c2 = st.columns(2)
    with c1:
        spec_p = full_fft.get("fft_spectrum_path")
        if spec_p and os.path.exists(spec_p):
            st.image(spec_p, caption="Full-Frame Frequency Pattern (2D-FFT)", use_container_width=True)
    with c2:
        rad_p = full_fft.get("fft_radial_plot_path")
        if rad_p and os.path.exists(rad_p):
            st.image(rad_p, caption="Full-Frame Detail Curve vs. Normal Camera Trend", use_container_width=True)

    st.write("")
    _render_verdict_and_details(
        fft_translation,
        verdict_title="Frequency (2D-FFT) Verdict",
        expander_label="📖 Why did it say this? (Simple explanation)",
    )

    with st.expander("📊 Technical 2D-FFT Metrics (For Experts)", expanded=False):
        if face_ffts:
            for f_fft in face_ffts:
                fm = f_fft.get("metrics", {})
                st.markdown(
                    f"**{f_fft['global_face_id']}** — BBox `{f_fft['bbox']}` | "
                    f"Score: `{f_fft['fft_score']:.4f} ({f_fft['fft_score'] * 100:.0f}%)`"
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
        st.markdown("**Full-Frame 2D-FFT Metrics**")
        st.table(
            {
                "Full-Frame FFT Metric": [
                    "Full-Frame FFT Score",
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
        with st.expander("📖 Detailed Breakdown for This Frame", expanded=False):
            for takeaway in takeaways:
                st.markdown(f"- {takeaway}")

    s = frame_m3["summary"]
    with st.expander("📊 Technical Frame Record & File Paths", expanded=False):
        st.table(
            {
                "Module 3 Property": [
                    "Frame Index",
                    "Source Raw Frame",
                    "Face Border & Color Score",
                    "Video Flicker Score",
                    "Pixel Grain (Noise) Score",
                    "Effective 2D-FFT Score",
                    "Combined Module 3 Score",
                    "Saved Module 3 Record (.json)",
                ],
                "Observed Value": [
                    str(frame_m3["frame_index"]),
                    frame_m3["frame_path"],
                    f"{s['boundary_score']:.4f} ({s['boundary_score'] * 100:.0f}%)"
                    if s.get("boundary_evaluated")
                    else "N/A (No face)",
                    f"{s['temporal_score']:.4f} ({s['temporal_score'] * 100:.0f}%)"
                    if s.get("temporal_applicable")
                    else "N/A (Single Image)",
                    f"{s['noise_score']:.4f} ({s['noise_score'] * 100:.0f}%)",
                    f"{s['effective_fft_score']:.4f} ({s['effective_fft_score'] * 100:.0f}%)",
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
    .status-file {
        font-size: 0.85rem;
        font-weight: 400;
        opacity: 0.78;
        margin-left: 6px;
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
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .m3-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
        border: 1px solid;
    }
    .pill-low {
        background-color: rgba(33, 195, 84, 0.16);
        border-color: rgba(33, 195, 84, 0.55);
        color: #7ee8a2;
    }
    .pill-med {
        background-color: rgba(255, 189, 46, 0.22);
        border-color: rgba(255, 189, 46, 0.75);
        color: #ffd97d;
    }
    .pill-high {
        background-color: rgba(255, 75, 75, 0.28);
        border-color: #ff4b4b;
        color: #ffb3b3;
        font-weight: 700;
    }
    .pill-na {
        background-color: rgba(150, 160, 175, 0.14);
        border-color: rgba(150, 160, 175, 0.40);
        color: #b0b8c4;
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
    }

st.title("🔍 Digital Media & Deepfake Check")

# =========================================================
# SIDEBAR: CONFIGURATION & INTEGRITY LOCK
# =========================================================
st.sidebar.header("⚙️ 1. Analysis Settings")

is_locked = st.session_state["settings_locked"]
current_cfg = st.session_state["applied_settings"]

sel_backend = st.sidebar.selectbox(
    "Face Finder Model",
    options=["retinaface", "mtcnn"],
    index=0 if current_cfg["detector_backend"] == "retinaface" else 1,
    disabled=is_locked,
    help="RetinaFace is more accurate at finding faces in busy scenes.",
)

sel_confidence = st.sidebar.slider(
    "Face Detection Strictness",
    min_value=0.50,
    max_value=0.99,
    value=float(current_cfg["min_confidence"]),
    step=0.01,
    disabled=is_locked,
    help="Higher values ignore blurry or uncertain background faces.",
)

sel_size = st.sidebar.selectbox(
    "Cropped Face Size",
    options=[299, 224],
    index=0 if current_cfg["output_size"] == 299 else 1,
    disabled=is_locked,
    help="299x299 matches the standard size expected by AI deepfake models (Module 4).",
)

sel_max_frames = st.sidebar.slider(
    "Frames to Check (Video)",
    min_value=2,
    max_value=30,
    value=int(current_cfg["max_frames"]),
    step=1,
    disabled=is_locked,
    help=(
        "How many evenly spaced frames to extract from a video. Blank/black frames are skipped and "
        "replaced automatically, and 5 consecutive frames are checked at each frame for flicker."
    ),
)

st.sidebar.markdown("---")
if not is_locked:
    if st.sidebar.button(
        "🔒 Lock Settings & Run", use_container_width=True, type="primary"
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
    st.sidebar.success("🔒 Settings are LOCKED for consistent results.")
    if st.sidebar.button("🔓 Unlock to Edit Settings", use_container_width=True):
        st.session_state["settings_locked"] = False
        st.session_state["m2_result"] = None
        st.session_state["m3_result"] = None
        st.session_state.pop("m2_cache_key", None)
        st.session_state.pop("m2_frame_record_paths", None)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📁 2. Upload Photo or Video")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video file",
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
    hashes = calculate_hashes(file_path)
    file_sha = hashes["SHA-256"] if hashes else upload_buf_hash

    # =====================================================
    # MODULE 1: EVIDENCE INGESTION & DATA INTEGRITY
    # =====================================================
    st.header("🟢 Module 1: Uploaded File & Digital Fingerprint")
    st.caption(f"🗄️ Saved working copy: `{file_path}`")
    col_ev1, col_ev2 = st.columns([1, 1])

    with col_ev1:
        st.subheader("Original File")
        if uploaded_file.name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            col_l, col_img, col_r = st.columns([0.1, 0.8, 0.1])
            with col_img:
                st.image(Image.open(file_path), caption=Path(file_path).name, width=420)
        else:
            play_path, _ = _get_browser_playable_video(file_path, file_sha)
            st.video(play_path)

    with col_ev2:
        st.subheader("File Verification Hashes")
        if hashes:
            st.info(f"**MD5:** `{hashes['MD5']}`\n\n**SHA-256:** `{hashes['SHA-256']}`")

        with st.expander("🔎 Hidden File Info (EXIF Metadata)", expanded=False):
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
    st.header("🟡 Module 2: Frame Extraction & Face Finding")

    if not is_locked:
        st.warning(
            "⚠️ **Paused:** Settings are currently **UNLOCKED**. "
            "Adjust the options on the left sidebar, then click **'🔒 Lock Settings & Run'** to analyze this file."
        )
    else:
        active_cfg = st.session_state["applied_settings"]
        cache_key = (
            f"{file_sha[:8]}_{active_cfg['detector_backend']}_{active_cfg['min_confidence']:.2f}_"
            f"{active_cfg['output_size']}_{active_cfg['max_frames']}_m3v8"
        )

        if st.session_state.get("m2_cache_key") != cache_key:
            with st.spinner(
                "Extracting frames, locating faces, and running Module 3 signal checks..."
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
                        burst_length=5,
                    )
                    frame_record_paths = _save_module2_json_records(m2_res)
                    m3_res = _run_module3_on_raw_frames(m2_res)
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
            k1.metric("File Type", m2_res["forensics_metadata"]["media_type"])
            k2.metric("Face Finder", m2_res["preprocessing_settings"]["ai_detector"])
            k3.metric("Frames Extracted", exec_sum["final_frames_analyzed"])
            k4.metric("Blank Frames Skipped", exec_sum["discarded_blank_or_extreme_frames"])
            k5.metric("Faces Cropped", exec_sum["total_faces_extracted"])

            st.caption(f"📁 Output folder: `{m2_res['artifacts_directory']['root']}`")

            # =================================================
            # MODULE 2 INTERACTIVE FRAME INSPECTION
            # =================================================
            total_frames = len(m2_res["frames_detail"])
            if total_frames > 0:
                st.markdown("---")
                if total_frames > 1:
                    col_sel1, col_sel2 = st.columns([2, 5])
                    with col_sel1:
                        st.markdown("#### 🎞️ Pick a Frame to View")
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
                    "👤 1. Cropped Face",
                    "🎯 2. Face Box & Points",
                    "🎞️ 3. Full Frame",
                    "📄 4. Frame Details",
                ]

                st.write("")
                if hasattr(st, "segmented_control"):
                    selected_mode = st.segmented_control(
                        "Module 2 View:",
                        options=mode_options,
                        default=mode_options[0],
                        key="m2_mode_selector",
                        label_visibility="collapsed",
                    )
                else:
                    selected_mode = st.radio(
                        "Module 2 View:",
                        options=mode_options,
                        index=0,
                        horizontal=True,
                        key="m2_mode_selector",
                        label_visibility="collapsed",
                    )

                st.write("")
                st.info(
                    f"**Viewing Frame:** #{selected_frame_idx} (`{Path(current_frame['frame_path']).name}`) | "
                    f"**Faces Found:** {current_frame['faces_detected']}"
                )

                if selected_mode == mode_options[0]:
                    faces = current_frame.get("faces", [])
                    if not faces:
                        st.warning("No clear face was found in this frame.")
                    else:
                        for f_info in faces:
                            c_img, c_info = st.columns([1, 2])
                            with c_img:
                                st.image(
                                    f_info["aligned_face_path"],
                                    caption=f"{f_info.get('global_face_id', 'Face')} ({f_info['confidence'] * 100:.0f}% match)",
                                    width=260,
                                )
                            with c_info:
                                st.markdown("##### 📌 Face Crop Info")
                                st.markdown(
                                    f"""
                                    - **Face ID:** `{f_info.get('global_face_id', 'Face')}`
                                    - **Match Confidence:** `{f_info['confidence'] * 100:.1f}%`
                                    - **Standardized Crop Size:** `{f_info['aligned_size'][0]} x {f_info['aligned_size'][1]} px`
                                    - **Face Location Box:** `[{f_info['bbox'][0]}, {f_info['bbox'][1]}, {f_info['bbox'][2]}, {f_info['bbox'][3]}]`
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
                                caption=f"Detected Face Box & 5 Key Points: {Path(ann_path).name}",
                                width=480,
                            )
                    else:
                        st.info("No annotated image available.")

                elif selected_mode == mode_options[2]:
                    col_l, col_center, col_r = st.columns([1, 2, 1])
                    with col_center:
                        st.image(
                            current_frame["frame_path"],
                            caption=f"Unedited Frame: {Path(current_frame['frame_path']).name}",
                            width=480,
                        )

                elif selected_mode == mode_options[3]:
                    _render_frame_forensic_record(
                        current_frame,
                        record_file_path=frame_record_paths.get(selected_frame_idx - 1),
                    )

            # =================================================
            # MODULE 2 SUMMARY
            # =================================================
            st.divider()
            st.subheader("📋 Frame Extraction Summary")
            with st.expander("🗂️ Technical Preprocessing Details", expanded=False):
                st.table(
                    {
                        "Property": [
                            "Case ID",
                            "Timestamp",
                            "Media Type",
                            "Face Finder",
                            "Sampling Mode",
                            "Cropped Face Size",
                            "Frames Extracted",
                            "Blank Frames Skipped",
                            "Total Faces Cropped",
                            "Output Folder",
                        ],
                        "Value": [
                            m2_res["forensics_metadata"]["case_id"],
                            m2_res["forensics_metadata"]["timestamp"],
                            m2_res["forensics_metadata"]["media_type"],
                            m2_res["preprocessing_settings"]["ai_detector"],
                            m2_res["preprocessing_settings"]["sampling_mode"],
                            m2_res["preprocessing_settings"]["canonical_resolution"],
                            str(exec_sum["final_frames_analyzed"]),
                            str(exec_sum["discarded_blank_or_extreme_frames"]),
                            str(exec_sum["total_faces_extracted"]),
                            m2_res["artifacts_directory"]["root"],
                        ],
                    }
                )

            # =================================================
            # MODULE 3: SIGNAL-LEVEL CHECKS
            # =================================================
            if m3_res:
                st.divider()
                st.header("🟠 Module 3: Physical Clues (Blending, Flicker, Grain & Frequency)")
                st.caption(
                    "Inspects unedited frames for face-border blending seams, video flicker across consecutive frames, "
                    "pixel-grain differences, and repeating AI frequency patterns."
                )

                m3_sum = m3_res["execution_summary"]
                frames_detail_m3 = m3_res.get("frames_detail", []) or []
                is_multi_frame = len(frames_detail_m3) > 1
                if is_multi_frame:
                    peak_frame_m3 = max(
                        frames_detail_m3,
                        key=lambda f: float(f["summary"]["combined_module3_score"]),
                    )
                    peak_pct_val = float(peak_frame_m3["summary"]["combined_module3_score"]) * 100.0
                    m3_top_subtitle = (
                        f"Frame #{peak_frame_m3['frame_index']} has the highest suspicion ({peak_pct_val:.0f}%)."
                    )
                elif len(frames_detail_m3) == 1:
                    m3_top_subtitle = (
                        f"Analyzed 1 frame ({Path(frames_detail_m3[0]['frame_path']).name})."
                    )
                else:
                    m3_top_subtitle = None

                _render_suspicion_box(
                    title="Module 3 Suspicion",
                    severity=m3_sum["overall_suspicion"],
                    score=m3_sum["max_module3_score"],
                    subtitle=m3_top_subtitle,
                )

                mk1, mk2, mk3, mk4, mk5 = st.columns(5)
                mk1.metric("Frames Checked", m3_sum["frames_analyzed"])
                _render_m3_kpi(
                    mk2,
                    "Face Border & Color",
                    m3_sum["mean_boundary_score"],
                    m3_sum["max_boundary_score"],
                    is_multi_frame,
                )
                if m3_sum.get("temporal_analyzed"):
                    _render_m3_kpi(
                        mk3,
                        "Video Flicker",
                        m3_sum["mean_temporal_score"],
                        m3_sum["max_temporal_score"],
                        is_multi_frame,
                    )
                else:
                    mk3.metric("Video Flicker", "Image Only")
                _render_m3_kpi(
                    mk4,
                    "Pixel Grain (Noise)",
                    m3_sum["mean_noise_score"],
                    m3_sum["max_noise_score"],
                    is_multi_frame,
                )
                _render_m3_kpi(
                    mk5,
                    "Frequency (2D-FFT)",
                    m3_sum["mean_fft_score"],
                    m3_sum["max_fft_score"],
                    is_multi_frame,
                )

                total_m3_frames = len(frames_detail_m3)
                if total_m3_frames > 0:
                    st.markdown("---")
                    if total_m3_frames > 1:
                        m3_col1, m3_col2 = st.columns([2, 5])
                        with m3_col1:
                            st.markdown("#### 🎞️ Pick a Frame to Inspect")
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

                    current_m3 = frames_detail_m3[m3_frame_idx - 1]
                    m3_modes = [
                        "🎭 1. Face Border & Color",
                        "🎞️ 2. Video Flicker",
                        "🔬 3. Pixel Grain (Noise)",
                        "📡 4. Frequency Scan (2D-FFT)",
                        "📄 5. Frame Summary",
                    ]

                    st.write("")
                    if hasattr(st, "segmented_control"):
                        m3_mode = st.segmented_control(
                            "Module 3 Check:",
                            options=m3_modes,
                            default=m3_modes[0],
                            key="m3_mode_selector",
                            label_visibility="collapsed",
                        )
                    else:
                        m3_mode = st.radio(
                            "Module 3 Check:",
                            options=m3_modes,
                            index=0,
                            horizontal=True,
                            key="m3_mode_selector",
                            label_visibility="collapsed",
                        )

                    _render_frame_status_box(m3_frame_idx, current_m3)

                    if m3_mode == m3_modes[0]:
                        _render_module3_boundary(
                            current_m3["boundary"],
                            boundary_translation=current_m3.get("interpretation", {}).get("boundary"),
                        )
                    elif m3_mode == m3_modes[1]:
                        _render_module3_temporal(
                            current_m3["temporal"],
                            temporal_translation=current_m3.get("interpretation", {}).get("temporal"),
                        )
                    elif m3_mode == m3_modes[2]:
                        _render_module3_noise(
                            current_m3["noise"],
                            noise_translation=current_m3.get("interpretation", {}).get("noise"),
                        )
                    elif m3_mode == m3_modes[3]:
                        _render_module3_fft(current_m3)
                    elif m3_mode == m3_modes[4]:
                        _render_module3_frame_record(current_m3)

                # =============================================
                # MODULE 3 AUDIT SUMMARY & OVERALL VERDICT
                # =============================================
                st.divider()
                st.subheader("📋 Module 3 Overall Summary")

                _render_verdict_and_details(
                    m3_res.get("interpretation"),
                    verdict_title="Overall Module 3 Verdict",
                    expander_label="📖 See frame-by-frame breakdown & notable frames",
                )

                with st.expander("🗂️ Technical Audit Metadata & Manifest Path", expanded=False):
                    st.table(
                        {
                            "Module 3 Property": [
                                "Case ID",
                                "Timestamp",
                                "Active Signals",
                                "Mean / Max Face Border Score",
                                "Mean / Max Video Flicker Score",
                                "Mean / Max Pixel Grain Score",
                                "Mean / Max Effective FFT Score",
                                "Mean / Max Module 3 Score",
                                "Overall Module 3 Suspicion",
                                "Module 3 Manifest File",
                            ],
                            "Observed Value": [
                                m3_res["forensics_metadata"]["case_id"],
                                m3_res["forensics_metadata"]["timestamp"],
                                ", ".join(m3_res["module3_settings"]["active_signals"]),
                                f"{m3_sum['mean_boundary_score']:.4f} / {m3_sum['max_boundary_score']:.4f}",
                                f"{m3_sum['mean_temporal_score']:.4f} / {m3_sum['max_temporal_score']:.4f}"
                                if m3_sum.get("temporal_analyzed")
                                else "N/A (Single Image)",
                                f"{m3_sum['mean_noise_score']:.4f} / {m3_sum['max_noise_score']:.4f}",
                                f"{m3_sum['mean_fft_score']:.4f} / {m3_sum['max_fft_score']:.4f}",
                                f"{m3_sum['mean_module3_score']:.4f} / {m3_sum['max_module3_score']:.4f}",
                                str(m3_sum["overall_suspicion"]).upper(),
                                m3_res["artifacts_directory"]["manifest"],
                            ],
                        }
                    )
else:
    st.info(
        "👈 Adjust settings on the left sidebar, click 'Lock Settings & Run', and upload a photo or video to begin."
    )
