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


def _is_clipped_or_extreme_profile(
    face: Dict[str, Any], img_w: int, img_h: int
) -> bool:
    bbox = face.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(x2 - x1, 1.0)

    if x1 < -2.0 or y1 < -2.0 or x2 > (img_w + 2.0) or y2 > (img_h + 2.0):
        return True

    lm = face.get("landmarks") or {}
    for pt in lm.values():
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            if pt[0] < 0 or pt[0] > img_w or pt[1] < 0 or pt[1] > img_h:
                return True

    le = lm.get("left_eye")
    re = lm.get("right_eye")
    nose = lm.get("nose")
    if le and re and nose:
        eye_dx = abs(float(re[0]) - float(le[0]))
        min_ex = min(float(le[0]), float(re[0]))
        max_ex = max(float(le[0]), float(re[0]))
        nose_x = float(nose[0])
        if eye_dx < 0.24 * bw or nose_x < (min_ex - 0.05 * bw) or nose_x > (max_ex + 0.05 * bw):
            return True

    return False


class TemporalAnalyzer:
    """
    Module 3.4: Multi-Face Temporal Face Flicker & Inter-Frame Consistency.
    Independently tracks EVERY face found on the anchor frame across the burst,
    generates side-by-side visual artifacts (Raw Crop, Flicker Heatmap, Stability Chart)
    per face, and matches global face IDs.
    """

    @staticmethod
    def _compute_suspicion_level(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def _analyze_single_track(
        self, tracked_series: List[Tuple[np.ndarray, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        crop_size = 160
        margin = 24

        lm_seq: List[np.ndarray] = []
        diag_seq: List[float] = []
        wh_seq: List[np.ndarray] = []

        aligned_cores_bgr: List[np.ndarray] = []
        core_masks: List[np.ndarray] = []
        lab_medians: List[Tuple[float, float, float]] = []

        anchor_pts = _landmarks_to_array(tracked_series[0][1])
        anchor_bbox = tracked_series[0][1]["bbox"]
        any_unreliable = False

        for img, face in tracked_series:
            ih, iw = img.shape[:2]
            if _is_clipped_or_extreme_profile(face, iw, ih):
                any_unreliable = True

            bbox = face["bbox"]
            bw = max(float(bbox[2]) - float(bbox[0]), 1.0)
            bh = max(float(bbox[3]) - float(bbox[1]), 1.0)
            diag = float(np.hypot(bw, bh))
            pts = _landmarks_to_array(face)

            if pts is not None:
                lm_seq.append(pts)
                diag_seq.append(diag)
                wh_seq.append(np.array([bw, bh], dtype=np.float32))

            src_mask = np.ones((ih, iw), dtype=np.uint8) * 255

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
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    w_mask = (
                        cv2.warpAffine(
                            src_mask,
                            mat,
                            (crop_size, crop_size),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        > 0
                    )
                else:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
                    crop = img[max(0, y1) : min(ih, y2), max(0, x1) : min(iw, x2)]
                    if crop.size == 0:
                        warped = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                        w_mask = np.zeros((crop_size, crop_size), dtype=bool)
                    else:
                        warped = cv2.resize(crop, (crop_size, crop_size))
                        w_mask = np.ones((crop_size, crop_size), dtype=bool)
            else:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
                crop = img[max(0, y1) : min(ih, y2), max(0, x1) : min(iw, x2)]
                if crop.size == 0:
                    warped = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    w_mask = np.zeros((crop_size, crop_size), dtype=bool)
                else:
                    warped = cv2.resize(crop, (crop_size, crop_size))
                    w_mask = np.ones((crop_size, crop_size), dtype=bool)

            core = warped[margin : crop_size - margin, margin : crop_size - margin]
            c_mask = w_mask[margin : crop_size - margin, margin : crop_size - margin]
            aligned_cores_bgr.append(core)
            core_masks.append(c_mask)

            lab_core = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).astype(np.float32)
            valid_px = lab_core[c_mask] if np.any(c_mask) else lab_core.reshape(-1, 3)
            lab_medians.append(
                (
                    float(np.median(valid_px[:, 0])),
                    float(np.median(valid_px[:, 1])) - 128.0,
                    float(np.median(valid_px[:, 2])) - 128.0,
                )
            )

        landmark_jitter = 0.0
        bbox_jitter = 0.0
        if len(lm_seq) >= 3:
            lm_arr = np.stack(lm_seq, axis=0)
            mean_diag = float(np.mean(diag_seq))
            accel = lm_arr[2:] - 2.0 * lm_arr[1:-1] + lm_arr[:-2]
            per_step_lm = np.mean(np.linalg.norm(accel, axis=2), axis=1) / max(mean_diag, 1.0)
            landmark_jitter = float(np.mean(per_step_lm))

            wh_arr = np.stack(wh_seq, axis=0)
            wh_accel = np.abs(wh_arr[2:] - 2.0 * wh_arr[1:-1] + wh_arr[:-2]) / np.maximum(
                wh_arr[1:-1], 1.0
            )
            bbox_jitter = float(np.mean(wh_accel))

        hp_cores: List[np.ndarray] = []
        for core in aligned_cores_bgr:
            g = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hp = cv2.filter2D(g, cv2.CV_32F, SRM_KV_KERNEL)
            hp_cores.append(hp)

        pair_flicker_ratios: List[float] = []
        flicker_accum = np.zeros_like(hp_cores[0], dtype=np.float32)
        for i in range(len(hp_cores) - 1):
            common_valid = core_masks[i] & core_masks[i + 1]
            diff = np.abs(hp_cores[i + 1] - hp_cores[i])
            if np.sum(common_valid) >= 16:
                diff_masked = np.where(common_valid, diff, 0.0)
                flicker_accum += diff_masked
                base_energy = 0.5 * (
                    float(np.mean(np.abs(hp_cores[i][common_valid])))
                    + float(np.mean(np.abs(hp_cores[i + 1][common_valid])))
                )
                pair_flicker_ratios.append(
                    float(np.mean(diff[common_valid])) / max(base_energy, 1.5)
                )
            else:
                flicker_accum += diff
                base_energy = 0.5 * (
                    float(np.mean(np.abs(hp_cores[i])))
                    + float(np.mean(np.abs(hp_cores[i + 1])))
                )
                pair_flicker_ratios.append(float(np.mean(diff)) / max(base_energy, 1.5))

        texture_flicker_ratio = (
            float(np.mean(pair_flicker_ratios)) if pair_flicker_ratios else 0.0
        )

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

        return {
            "score": temporal_score,
            "suspicion_level": self._compute_suspicion_level(temporal_score),
            "unreliable": any_unreliable,
            "tracked_frames": len(tracked_series),
            "landmark_jitter": round(landmark_jitter, 5),
            "bbox_jitter": round(bbox_jitter, 5),
            "texture_flicker_ratio": round(texture_flicker_ratio, 4),
            "color_flicker_lab": round(color_flicker_lab, 4),
            "brightness_flicker": round(brightness_flicker, 4),
            "flicker_accum": flicker_accum,
            "base_core": aligned_cores_bgr[0],
            "l_arr": l_arr,
            "a_arr": a_arr,
            "b_arr": b_arr,
            "pair_flicker_ratios": pair_flicker_ratios,
            "num_pairs": max(len(hp_cores) - 1, 1),
            "seed_bbox": tracked_series[0][1]["bbox"],
            "seed_img": tracked_series[0][0],
        }

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
                    "faces_tracked_count": 0,
                    "peak_face_id": None,
                    "peak_face_index": None,
                    "unreliable_face_warning": False,
                },
                "faces": [],
                "untracked_faces": [],
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "Temporal consistency requires consecutive video frames (not applicable to single images).",
            }

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
                    "faces_tracked_count": 0,
                    "peak_face_id": None,
                    "peak_face_index": None,
                    "unreliable_face_warning": False,
                },
                "faces": [],
                "untracked_faces": [],
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "Fewer than 2 readable frames in burst.",
            }

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
                    "faces_tracked_count": 0,
                    "peak_face_id": None,
                    "peak_face_index": None,
                    "unreliable_face_warning": False,
                },
                "faces": [],
                "untracked_faces": [],
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "No faces detected across this consecutive frame burst.",
            }

        seed_faces = sorted(
            loaded[seed_idx][1],
            key=lambda f: int(f.get("face_index", 0)),
        )

        used_per_frame: Dict[int, set[int]] = {t: set() for t in range(len(loaded))}
        analyzed_faces: List[Dict[str, Any]] = []
        untracked_faces: List[Dict[str, Any]] = []

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)
        tag = prefix or "burst"

        for seed_face in seed_faces:
            f_idx = int(seed_face.get("face_index", len(analyzed_faces) + len(untracked_faces)))
            g_id = seed_face.get("global_face_id") or f"Face #{f_idx}"
            clean_gid = str(g_id).replace(" ", "_").replace("#", "")

            tracked_series: List[Tuple[np.ndarray, Dict[str, Any]]] = []
            prev_bbox = seed_face["bbox"]

            for t in range(seed_idx, len(loaded)):
                img_t, faces_t = loaded[t]
                if not faces_t:
                    continue

                candidates = [
                    (cand_i, cand_f, _bbox_iou(prev_bbox, cand_f["bbox"]))
                    for cand_i, cand_f in enumerate(faces_t)
                    if cand_i not in used_per_frame[t]
                ]
                if not candidates:
                    continue

                best_i, best_f, best_iou = max(candidates, key=lambda x: x[2])
                if best_iou >= 0.25:
                    used_per_frame[t].add(best_i)
                    tracked_series.append((img_t, best_f))
                    prev_bbox = best_f["bbox"]

            if len(tracked_series) < 2:
                untracked_faces.append(
                    {
                        "face_index": f_idx,
                        "global_face_id": g_id,
                        "tracked_frames": len(tracked_series),
                        "total_burst_frames": len(loaded),
                        "reason": (
                            "Appeared in only 1 frame of this burst and could not be tracked"
                            if len(tracked_series) == 1
                            else "Could not be tracked across consecutive frames in this burst"
                        ),
                    }
                )
                continue

            res = self._analyze_single_track(tracked_series)

            raw_crop_path_str: Optional[str] = None
            heat_path_str: Optional[str] = None
            plot_path_str: Optional[str] = None

            if out_path:
                # 1. Save reference raw crop from anchor frame
                s_img = res["seed_img"]
                s_box = res["seed_bbox"]
                ih, iw = s_img.shape[:2]
                bx1 = max(0, int(round(float(s_box[0]))))
                by1 = max(0, int(round(float(s_box[1]))))
                bx2 = min(iw, int(round(float(s_box[2]))))
                by2 = min(ih, int(round(float(s_box[3]))))
                raw_crop = s_img[by1:by2, bx1:bx2]
                if raw_crop.size > 0:
                    rc_file = out_path / f"{tag}_{clean_gid}_raw_crop.png"
                    _imwrite_unicode(rc_file, raw_crop)
                    raw_crop_path_str = str(rc_file.resolve())

                # 2. Save face skin flicker heatmap
                avg_flicker = res["flicker_accum"] / res["num_pairs"]
                p99 = float(np.percentile(avg_flicker, 99.0))
                norm = np.clip(avg_flicker / max(p99, 1.5), 0.0, 1.0)
                norm_u8 = np.clip(np.power(norm, 0.80) * 255.0, 0, 255).astype(np.uint8)
                heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
                overlay_core = cv2.addWeighted(res["base_core"], 0.45, heat, 0.55, 0.0)
                overlay_vis = cv2.resize(overlay_core, (240, 240), interpolation=cv2.INTER_CUBIC)

                hm_file = out_path / f"{tag}_{clean_gid}_temporal_flicker.png"
                _imwrite_unicode(hm_file, overlay_vis)
                heat_path_str = str(hm_file.resolve())

                # 3. Save face stability plot
                pl_file = out_path / f"{tag}_{clean_gid}_temporal_plot.png"
                fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=120)
                frames_x = np.arange(1, len(res["l_arr"]) + 1)

                axes[0].plot(
                    frames_x,
                    res["l_arr"] - res["l_arr"][0],
                    marker="o",
                    color="#ff9800",
                    label="Brightness Shift",
                )
                axes[0].plot(
                    frames_x,
                    res["a_arr"] - res["a_arr"][0],
                    marker="s",
                    color="#e91e63",
                    label="Red-Green Tint Shift",
                )
                axes[0].plot(
                    frames_x,
                    res["b_arr"] - res["b_arr"][0],
                    marker="^",
                    color="#2196f3",
                    label="Blue-Yellow Tint Shift",
                )
                axes[0].axhline(0.0, color="#607d8b", linestyle="--", linewidth=0.8)
                axes[0].set_title(f"{g_id} Color & Brightness Stability")
                axes[0].set_xlabel("Consecutive Burst Frame")
                axes[0].set_ylabel("Shift vs. Frame 1")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(fontsize=7)

                pair_x = np.arange(1, len(res["pair_flicker_ratios"]) + 1)
                axes[1].bar(
                    pair_x,
                    res["pair_flicker_ratios"],
                    color="#ff5722",
                    alpha=0.85,
                    label="Texture Flicker",
                )
                axes[1].axhline(
                    0.85, color="#607d8b", linestyle="--", linewidth=1.0, label="Normal Ceiling"
                )
                axes[1].set_title(f"{g_id} Frame-to-Frame Texture Flicker")
                axes[1].set_xlabel("Consecutive Frame Transition")
                axes[1].set_ylabel("Flicker Ratio")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=7)

                fig.tight_layout()
                fig.savefig(str(pl_file))
                plt.close(fig)
                plot_path_str = str(pl_file.resolve())

            face_record = {
                "face_index": f_idx,
                "global_face_id": g_id,
                "temporal_score": res["score"],
                "suspicion_level": res["suspicion_level"],
                "tracked_frames": res["tracked_frames"],
                "total_burst_frames": len(loaded),
                "unreliable": res["unreliable"],
                "metrics": {
                    "landmark_jitter": res["landmark_jitter"],
                    "bbox_jitter": res["bbox_jitter"],
                    "texture_flicker_ratio": res["texture_flicker_ratio"],
                    "color_flicker_lab": res["color_flicker_lab"],
                    "brightness_flicker": res["brightness_flicker"],
                },
                "raw_crop_path": raw_crop_path_str,
                "temporal_heatmap_path": heat_path_str,
                "temporal_plot_path": plot_path_str,
            }
            analyzed_faces.append(face_record)

        if not analyzed_faces:
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
                    "faces_tracked_count": 0,
                    "peak_face_id": None,
                    "peak_face_index": None,
                    "unreliable_face_warning": False,
                },
                "faces": [],
                "untracked_faces": untracked_faces,
                "temporal_heatmap_path": None,
                "temporal_plot_path": None,
                "note": "Faces could not be tracked across consecutive frames in this burst.",
            }

        peak_face = max(analyzed_faces, key=lambda f: float(f["temporal_score"]))
        peak_score = float(peak_face["temporal_score"])

        return {
            "applicable": True,
            "burst_frames_count": len(loaded),
            "temporal_score": peak_score,
            "suspicion_level": self._compute_suspicion_level(peak_score),
            "metrics": {
                "landmark_jitter": peak_face["metrics"]["landmark_jitter"],
                "bbox_jitter": peak_face["metrics"]["bbox_jitter"],
                "texture_flicker_ratio": peak_face["metrics"]["texture_flicker_ratio"],
                "color_flicker_lab": peak_face["metrics"]["color_flicker_lab"],
                "brightness_flicker": peak_face["metrics"]["brightness_flicker"],
                "tracked_frames": peak_face["tracked_frames"],
                "faces_tracked_count": len(analyzed_faces),
                "peak_face_id": peak_face["global_face_id"],
                "peak_face_index": peak_face["face_index"],
                "unreliable_face_warning": peak_face["unreliable"],
            },
            "faces": analyzed_faces,
            "untracked_faces": untracked_faces,
            # Kept for backward compatibility with older components
            "temporal_heatmap_path": peak_face["temporal_heatmap_path"],
            "temporal_plot_path": peak_face["temporal_plot_path"],
        }
