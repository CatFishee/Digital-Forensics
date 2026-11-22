from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRM_KV_KERNEL = (
    np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float32,
    )
    / 12.0
)


def _imread_bgr_unicode(path: Path) -> Optional[np.ndarray]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def _landmarks_to_array(face: Dict[str, Any]) -> Optional[np.ndarray]:
    lm = face.get("landmarks") or {}
    keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    pts = []
    for k in keys:
        pt = lm.get(k)
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        pts.append([float(pt[0]), float(pt[1])])
    return np.asarray(pts, dtype=np.float32)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-6))


class TemporalAnalyzer:
    """
    Module 3.4: Temporal Face Flicker & Inter-Frame Consistency (Video Bursts).
    Analyzes short consecutive frame bursts around each sampled video moment.
    Measures:
      1. 2nd-order facial landmark & bounding-box jitter (ignores smooth real head motion).
      2. Motion-compensated inner-face texture flicker (via landmark affine alignment).
      3. Frame-to-frame skin brightness & color popping (2nd-order CIE Lab stability).
    """

    @staticmethod
    def _compute_suspicion_level(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def analyze(
        self,
        burst_frames: Sequence[Dict[str, Any]],
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not burst_frames or len(burst_frames) < 2:
            return {
                "applicable": False,
                "burst_frames_count": len(burst_frames or []),
                "temporal_score": 0.0,
                "suspicion_level": "low",
                "metrics": {
                    "landmark_jitter": None,
                    "bbox_jitter": None,
                    "texture_flicker_ratio": None,
                    "color_flicker_lab": None,
                    "brightness_flicker": None,
                    "tracked_frames": len(burst_frames or []),
                },
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "Temporal consistency requires consecutive video frames (not applicable to single images).",
            }

        # Load consecutive burst images
        loaded: List[Tuple[np.ndarray, List[Dict[str, Any]]]] = []
        for item in burst_frames:
            fp = Path(item["frame_path"])
            img = _imread_bgr_unicode(fp)
            if img is not None:
                loaded.append((img, list(item.get("faces", []) or [])))

        if len(loaded) < 2:
            return {
                "applicable": False,
                "burst_frames_count": len(loaded),
                "temporal_score": 0.0,
                "suspicion_level": "low",
                "metrics": {
                    "landmark_jitter": None,
                    "bbox_jitter": None,
                    "texture_flicker_ratio": None,
                    "color_flicker_lab": None,
                    "brightness_flicker": None,
                    "tracked_frames": len(loaded),
                },
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "Fewer than 2 readable frames in burst.",
            }

        # Pick primary face from the first frame in the burst that has a face
        seed_idx = next((i for i, (_, fs) in enumerate(loaded) if fs), None)
        if seed_idx is None:
            return {
                "applicable": True,
                "burst_frames_count": len(loaded),
                "temporal_score": 0.0,
                "suspicion_level": "low",
                "metrics": {
                    "landmark_jitter": None,
                    "bbox_jitter": None,
                    "texture_flicker_ratio": None,
                    "color_flicker_lab": None,
                    "brightness_flicker": None,
                    "tracked_frames": 0,
                },
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "No faces detected across this consecutive frame burst.",
            }

        # Track the primary face across the burst via IoU matching
        ref_face = max(
            loaded[seed_idx][1],
            key=lambda f: (float(f["bbox"][2]) - float(f["bbox"][0]))
            * (float(f["bbox"][3]) - float(f["bbox"][1])),
        )
        ref_bbox = ref_face["bbox"]

        tracked_series: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        prev_bbox = ref_bbox
        for img, faces in loaded:
            if not faces:
                continue
            best_face = max(faces, key=lambda f: _bbox_iou(prev_bbox, f["bbox"]))
            if _bbox_iou(prev_bbox, best_face["bbox"]) >= 0.25:
                tracked_series.append((img, best_face))
                prev_bbox = best_face["bbox"]

        if len(tracked_series) < 2:
            return {
                "applicable": True,
                "burst_frames_count": len(loaded),
                "temporal_score": 0.0,
                "suspicion_level": "low",
                "metrics": {
                    "landmark_jitter": None,
                    "bbox_jitter": None,
                    "texture_flicker_ratio": None,
                    "color_flicker_lab": None,
                    "brightness_flicker": None,
                    "tracked_frames": len(tracked_series),
                },
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "Face could not be tracked across consecutive frames in this burst.",
            }

        # 1. Landmark & BBox 2nd-order jitter (acceleration — ignores smooth head motion)
        lm_seq: List[np.ndarray] = []
        diag_seq: List[float] = []
        wh_seq: List[np.ndarray] = []

        # Align all tracked face crops into a canonical 160x160 interior core for texture/color flicker
        crop_size = 160
        margin = 24  # analyze inner 112x112 core away from background borders
        aligned_cores_bgr: List[np.ndarray] = []
        lab_medians: List[Tuple[float, float, float]] = []

        # Reference destination points inside 160x160 frame
        anchor_pts = _landmarks_to_array(tracked_series[0][1])
        anchor_bbox = tracked_series[0][1]["bbox"]

        for img, face in tracked_series:
            bbox = face["bbox"]
            bw = max(float(bbox[2]) - float(bbox[0]), 1.0)
            bh = max(float(bbox[3]) - float(bbox[1]), 1.0)
            diag = float(np.hypot(bw, bh))
            pts = _landmarks_to_array(face)

            if pts is not None:
                lm_seq.append(pts)
                diag_seq.append(diag)
                wh_seq.append(np.array([bw, bh], dtype=np.float32))

            # Stabilize face patch onto common grid
            if pts is not None and anchor_pts is not None:
                ax1, ay1, ax2, ay2 = [float(v) for v in anchor_bbox[:4]]
                abw = max(ax2 - ax1, 1.0)
                abh = max(ay2 - ay1, 1.0)
                dst_pts = anchor_pts.copy()
                dst_pts[:, 0] = (dst_pts[:, 0] - ax1) / abw * crop_size
                dst_pts[:, 1] = (dst_pts[:, 1] - ay1) / abh * crop_size

                mat, _ = cv2.estimateAffinePartial2D(pts, dst_pts, method=cv2.LMEDS)
                if mat is not None:
                    warped = cv2.warpAffine(
                        img,
                        mat,
                        (crop_size, crop_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                else:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
                    crop = img[max(0, y1) : min(img.shape[0], y2), max(0, x1) : min(img.shape[1], x2)]
                    warped = cv2.resize(crop, (crop_size, crop_size))
            else:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
                crop = img[max(0, y1) : min(img.shape[0], y2), max(0, x1) : min(img.shape[1], x2)]
                warped = cv2.resize(crop, (crop_size, crop_size))

            core = warped[margin : crop_size - margin, margin : crop_size - margin]
            aligned_cores_bgr.append(core)

            lab_core = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_medians.append(
                (
                    float(np.median(lab_core[:, :, 0])),
                    float(np.median(lab_core[:, :, 1])) - 128.0,
                    float(np.median(lab_core[:, :, 2])) - 128.0,
                )
            )

        landmark_jitter = 0.0
        bbox_jitter = 0.0
        if len(lm_seq) >= 3:
            lm_arr = np.stack(lm_seq, axis=0)  # (T, 5, 2)
            mean_diag = float(np.mean(diag_seq))
            accel = lm_arr[2:] - 2.0 * lm_arr[1:-1] + lm_arr[:-2]
            per_step_lm = np.mean(np.linalg.norm(accel, axis=2), axis=1) / max(mean_diag, 1.0)
            landmark_jitter = float(np.mean(per_step_lm))

            wh_arr = np.stack(wh_seq, axis=0)
            wh_accel = np.abs(wh_arr[2:] - 2.0 * wh_arr[1:-1] + wh_arr[:-2]) / np.maximum(
                wh_arr[1:-1], 1.0
            )
            bbox_jitter = float(np.mean(wh_accel))

        # 2. Motion-compensated high-pass texture flicker
        hp_cores: List[np.ndarray] = []
        for core in aligned_cores_bgr:
            g = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hp = cv2.filter2D(g, cv2.CV_32F, SRM_KV_KERNEL)
            hp_cores.append(hp)

        pair_flicker_ratios: List[float] = []
        flicker_accum = np.zeros_like(hp_cores[0], dtype=np.float32)
        for i in range(len(hp_cores) - 1):
            diff = np.abs(hp_cores[i + 1] - hp_cores[i])
            flicker_accum += diff
            base_energy = 0.5 * (
                float(np.mean(np.abs(hp_cores[i])))
                + float(np.mean(np.abs(hp_cores[i + 1])))
            )
            pair_flicker_ratios.append(float(np.mean(diff)) / max(base_energy, 1.5))

        texture_flicker_ratio = float(np.mean(pair_flicker_ratios)) if pair_flicker_ratios else 0.0

        # 3. 2nd-order brightness & color popping inside face core
        brightness_flicker = 0.0
        color_flicker_lab = 0.0
        l_arr = np.array([v[0] for v in lab_medians], dtype=np.float32)
        a_arr = np.array([v[1] for v in lab_medians], dtype=np.float32)
        b_arr = np.array([v[2] for v in lab_medians], dtype=np.float32)

        if len(lab_medians) >= 3:
            l_acc = np.abs(l_arr[2:] - 2.0 * l_arr[1:-1] + l_arr[:-2])
            a_acc = a_arr[2:] - 2.0 * a_arr[1:-1] + a_arr[:-2]
            b_acc = b_arr[2:] - 2.0 * b_arr[1:-1] + b_arr[:-2]
            brightness_flicker = float(np.mean(l_acc))
            color_flicker_lab = float(np.mean(np.hypot(a_acc, b_acc)))
        else:
            brightness_flicker = float(np.mean(np.abs(np.diff(l_arr))))
            color_flicker_lab = float(
                np.mean(np.hypot(np.diff(a_arr), np.diff(b_arr)))
            )

        # 4. Calibrated Temporal Suspicion Score [0.0, 1.0]
        lm_term = float(np.clip((landmark_jitter - 0.014) / 0.028, 0.0, 1.0))
        bbox_term = float(np.clip((bbox_jitter - 0.022) / 0.045, 0.0, 1.0))
        tex_term = float(np.clip((texture_flicker_ratio - 0.85) / 0.55, 0.0, 1.0))
        lum_term = float(np.clip((brightness_flicker - 2.2) / 4.5, 0.0, 1.0))
        col_term = float(np.clip((color_flicker_lab - 1.4) / 3.2, 0.0, 1.0))

        subscores = [lm_term, bbox_term, tex_term, lum_term, col_term]
        temporal_score = round(
            float(
                np.clip(
                    0.55 * max(subscores) + 0.45 * float(np.mean(subscores)),
                    0.0,
                    1.0,
                )
            ),
            4,
        )

        # 5. Export XAI Heatmap & Burst Stability Timeline Plot
        heatmap_path: Optional[str] = None
        plot_path: Optional[str] = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            tag = prefix or "burst"

            avg_flicker = flicker_accum / max(len(hp_cores) - 1, 1)
            p99 = float(np.percentile(avg_flicker, 99.0))
            norm = np.clip(avg_flicker / max(p99, 1.5), 0.0, 1.0)
            norm_u8 = np.clip(np.power(norm, 0.80) * 255.0, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
            overlay_core = cv2.addWeighted(aligned_cores_bgr[0], 0.45, heat, 0.55, 0.0)
            overlay_vis = cv2.resize(overlay_core, (240, 240), interpolation=cv2.INTER_CUBIC)

            hm_file = out_path / f"{tag}_temporal_flicker.png"
            _imwrite_unicode(hm_file, overlay_vis)
            heatmap_path = str(hm_file.resolve())

            pl_file = out_path / f"{tag}_temporal_plot.png"
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=120)
            frames_x = np.arange(1, len(aligned_cores_bgr) + 1)

            axes[0].plot(frames_x, l_arr - l_arr[0], marker="o", color="#ff9800", label="Brightness Shift")
            axes[0].plot(frames_x, a_arr - a_arr[0], marker="s", color="#e91e63", label="Red-Green Tint Shift")
            axes[0].plot(frames_x, b_arr - b_arr[0], marker="^", color="#2196f3", label="Blue-Yellow Tint Shift")
            axes[0].axhline(0.0, color="#607d8b", linestyle="--", linewidth=0.8)
            axes[0].set_title("Face Color & Brightness Stability")
            axes[0].set_xlabel("Consecutive Burst Frame")
            axes[0].set_ylabel("Shift vs. Frame 1")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=7)

            pair_x = np.arange(1, len(pair_flicker_ratios) + 1)
            axes[1].bar(pair_x, pair_flicker_ratios, color="#ff5722", alpha=0.85, label="Texture Flicker")
            axes[1].axhline(0.85, color="#607d8b", linestyle="--", linewidth=1.0, label="Normal Ceiling")
            axes[1].set_title("Frame-to-Frame Texture Flicker")
            axes[1].set_xlabel("Consecutive Frame Transition")
            axes[1].set_ylabel("Flicker Ratio")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=7)

            fig.tight_layout()
            fig.savefig(str(pl_file))
            plt.close(fig)
            plot_path = str(pl_file.resolve())

        return {
            "applicable": True,
            "burst_frames_count": len(loaded),
            "temporal_score": temporal_score,
            "suspicion_level": self._compute_suspicion_level(temporal_score),
            "metrics": {
                "landmark_jitter": round(landmark_jitter, 5),
                "bbox_jitter": round(bbox_jitter, 5),
                "texture_flicker_ratio": round(texture_flicker_ratio, 4),
                "color_flicker_lab": round(color_flicker_lab, 4),
                "brightness_flicker": round(brightness_flicker, 4),
                "tracked_frames": len(tracked_series),
            },
            "temporal_heatmap_path": heatmap_path,
            "temporal_plot_path": plot_path,
        }
