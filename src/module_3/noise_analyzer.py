from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

# Classic 5x5 SRM (Steganalysis Rich Model) edge-suppressed noise residual kernel
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


def _compute_content_mask(gray: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """
    Exclude solid uniform outer borders (e.g., white/black letterbox margins)
    so framing does not bias background noise or block statistics.
    """
    h, w = gray.shape
    mask = np.ones((h, w), dtype=bool)
    if h < 16 or w < 16:
        return mask

    row_std = np.std(gray, axis=1)
    col_std = np.std(gray, axis=0)

    top = 0
    while top < h // 4 and row_std[top] < tol:
        top += 1
    bottom = h
    while bottom > (3 * h) // 4 and row_std[bottom - 1] < tol:
        bottom -= 1

    left = 0
    while left < w // 4 and col_std[left] < tol:
        left += 1
    right = w
    while right > (3 * w) // 4 and col_std[right - 1] < tol:
        right -= 1

    if bottom - top >= 16 and right - left >= 16:
        mask[:, :] = False
        mask[top:bottom, left:right] = True

    return mask


class NoiseAnalyzer:
    """
    Module 3.1: Calibrated SRM Noise Residual & Face-Boundary Blending Analysis.
    Works natively on BOTH extracted video PNG frames and JPEG images.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = max(8, int(block_size))

    @staticmethod
    def _compute_suspicion_level(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def analyze(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        face_bboxes: Optional[Sequence[Sequence[float]]] = None,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image file not found for Noise Analysis: {src}")

        img_bgr = _imread_bgr_unicode(src)
        if img_bgr is None:
            raise ValueError(f"Unable to read image for Noise Analysis: {src}")

        h, w = img_bgr.shape[:2]
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y_chan = ycrcb[:, :, 0]
        cr_chan = ycrcb[:, :, 1]
        cb_chan = ycrcb[:, :, 2]

        content_mask = _compute_content_mask(y_chan)

        # 1. SRM high-pass residuals on Luminance (Y) and Chrominance (Cr/Cb)
        y_res = np.abs(cv2.filter2D(y_chan, cv2.CV_32F, SRM_KV_KERNEL))
        cr_res = np.abs(cv2.filter2D(cr_chan, cv2.CV_32F, SRM_KV_KERNEL))
        cb_res = np.abs(cv2.filter2D(cb_chan, cv2.CV_32F, SRM_KV_KERNEL))
        chroma_res = 0.5 * (cr_res + cb_res)

        # 2. Block-level MEDIAN noise map (robust against sharp hair/text edges)
        bs = min(self.block_size, h, w)
        bh = max(1, h // bs)
        bw = max(1, w // bs)

        y_crop = y_res[: bh * bs, : bw * bs]
        c_crop = content_mask[: bh * bs, : bw * bs]

        y_blocks = np.median(y_crop.reshape(bh, bs, bw, bs), axis=(1, 3))
        valid_blocks = c_crop.reshape(bh, bs, bw, bs).mean(axis=(1, 3)) >= 0.5
        active_block_vals = (
            y_blocks[valid_blocks] if np.any(valid_blocks) else y_blocks.ravel()
        )

        # Quantization-safe denominators prevent flat H.264 background walls from blowing up ratios
        p50 = float(np.percentile(active_block_vals, 50))
        p95 = float(np.percentile(active_block_vals, 95))
        noise_p95_p50_ratio = float(p95 / max(p50, 1.0))
        noise_block_cv = float(
            np.std(active_block_vals) / max(float(np.mean(active_block_vals)), 0.90)
        )

        # 3. Face-vs-Background Skin Noise & Face-Boundary Seam Analysis
        face_noise_median: Optional[float] = None
        bg_noise_median: Optional[float] = None
        face_bg_noise_contrast: Optional[float] = None
        face_smoother_than_bg: Optional[bool] = None

        face_chroma_noise: Optional[float] = None
        bg_chroma_noise: Optional[float] = None
        face_bg_chroma_contrast: Optional[float] = None
        max_boundary_seam_ratio: Optional[float] = None

        if face_bboxes:
            face_mask = np.zeros((h, w), dtype=bool)
            seam_ratios = []

            for box in face_bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                bw_box = x2 - x1
                bh_box = y2 - y1
                if bw_box < 24 or bh_box < 24:
                    continue

                face_mask[y1:y2, x1:x2] = True

                # Compare SRM noise median along face perimeter vs inner core & outer context
                pad_x = max(4, int(round(0.16 * bw_box)))
                pad_y = max(4, int(round(0.16 * bh_box)))
                ox1, oy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                ox2, oy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                in_x = max(2, int(round(0.14 * bw_box)))
                in_y = max(2, int(round(0.14 * bh_box)))
                ix1, iy1 = min(x2, x1 + in_x), min(y2, y1 + in_y)
                ix2, iy2 = max(x1, x2 - in_x), max(y1, y2 - in_y)

                seam_t = max(2, int(round(0.05 * min(bw_box, bh_box))))
                seam_mask = np.zeros((h, w), dtype=bool)
                sy1_out, sy2_out = max(0, y1 - seam_t), min(h, y2 + seam_t)
                sx1_out, sx2_out = max(0, x1 - seam_t), min(w, x2 + seam_t)
                seam_mask[sy1_out:sy2_out, sx1_out:sx2_out] = True
                if (x2 - x1) > 2 * seam_t and (y2 - y1) > 2 * seam_t:
                    seam_mask[y1 + seam_t : y2 - seam_t, x1 + seam_t : x2 - seam_t] = False
                seam_mask &= content_mask

                core_mask = np.zeros((h, w), dtype=bool)
                if ix2 > ix1 and iy2 > iy1:
                    core_mask[iy1:iy2, ix1:ix2] = True
                core_mask &= content_mask

                outer_mask = np.zeros((h, w), dtype=bool)
                outer_mask[oy1:oy2, ox1:ox2] = True
                outer_mask[sy1_out:sy2_out, sx1_out:sx2_out] = False
                outer_mask &= content_mask

                if np.any(seam_mask) and np.any(core_mask) and np.any(outer_mask):
                    seam_energy = float(np.median(y_res[seam_mask]))
                    ref_energy = 0.5 * (
                        float(np.median(y_res[core_mask]))
                        + float(np.median(y_res[outer_mask]))
                    )
                    seam_ratios.append(seam_energy / max(ref_energy, 0.85))

            if seam_ratios:
                max_boundary_seam_ratio = float(max(seam_ratios))

            valid_face = face_mask & content_mask
            valid_bg = (~face_mask) & content_mask

            if np.any(valid_face) and np.any(valid_bg):
                # 35th percentile inside face isolates smooth skin noise away from eyes/brows/lips
                face_noise_median = float(np.percentile(y_res[valid_face], 35))
                bg_noise_median = float(np.median(y_res[valid_bg]))
                face_smoother_than_bg = face_noise_median < bg_noise_median
                face_bg_noise_contrast = float(
                    abs(face_noise_median - bg_noise_median)
                    / max(min(face_noise_median, bg_noise_median), 0.90)
                )

                face_chroma_noise = float(np.percentile(chroma_res[valid_face], 35))
                bg_chroma_noise = float(np.median(chroma_res[valid_bg]))
                face_bg_chroma_contrast = float(
                    abs(face_chroma_noise - bg_chroma_noise)
                    / max(min(face_chroma_noise, bg_chroma_noise), 0.40)
                )

        # 4. Calibrated Noise & Boundary Suspicion Score [0.0, 1.0]
        inconsistency_term = float(
            1.0 - np.exp(-0.22 * max(noise_p95_p50_ratio - 4.5, 0.0))
        )
        cv_term = float(np.clip((noise_block_cv - 0.95) / 1.40, 0.0, 1.0))
        global_subscore = float(
            np.clip(0.60 * inconsistency_term + 0.40 * cv_term, 0.0, 1.0)
        )

        subscores = [global_subscore]
        if face_bg_noise_contrast is not None:
            face_noise_term = float(
                1.0 - np.exp(-0.85 * max(face_bg_noise_contrast - 0.40, 0.0))
            )
            subscores.append(face_noise_term)

        if face_bg_chroma_contrast is not None:
            chroma_term = float(
                1.0 - np.exp(-0.85 * max(face_bg_chroma_contrast - 0.35, 0.0))
            )
            subscores.append(chroma_term)

        if max_boundary_seam_ratio is not None:
            seam_term = float(
                np.clip((max_boundary_seam_ratio - 1.65) / 1.35, 0.0, 1.0)
            )
            subscores.append(seam_term)

        noise_score = round(
            float(
                np.clip(
                    0.55 * float(np.max(subscores)) + 0.45 * float(np.mean(subscores)),
                    0.0,
                    1.0,
                )
            ),
            4,
        )

        # 5. Export XAI Visuals (SRM Residual Map + Smooth Block Variance Heatmap)
        residual_path: Optional[str] = None
        heatmap_path: Optional[str] = None

        if output_dir:
            out_path = Path(output_dir)
            tag = prefix or src.stem

            p99 = float(
                np.percentile(
                    y_res[content_mask] if np.any(content_mask) else y_res, 99.0
                )
            )
            scale = 255.0 / max(p99, 1.0)
            res_vis = np.clip(
                np.power(np.clip((y_res * scale) / 255.0, 0.0, 1.0), 0.75) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            res_bgr = cv2.cvtColor(res_vis, cv2.COLOR_GRAY2BGR)

            b_p98 = float(np.percentile(active_block_vals, 98.0))
            b_p02 = float(np.percentile(active_block_vals, 2.0))
            b_norm = np.clip(
                (y_blocks - b_p02) / max(b_p98 - b_p02, 1e-3), 0.0, 1.0
            )
            b_u8 = (b_norm * 255.0).astype(np.uint8)
            heat_gray = cv2.resize(b_u8, (w, h), interpolation=cv2.INTER_CUBIC)
            heat_bgr = cv2.applyColorMap(heat_gray, cv2.COLORMAP_TURBO)

            res_file = out_path / f"{tag}_noise_residual.png"
            heat_file = out_path / f"{tag}_noise_heatmap.png"
            _imwrite_unicode(res_file, res_bgr)
            _imwrite_unicode(heat_file, heat_bgr)

            residual_path = str(res_file.resolve())
            heatmap_path = str(heat_file.resolve())

        return {
            "source_path": str(src.resolve()),
            "noise_score": noise_score,
            "suspicion_level": self._compute_suspicion_level(noise_score),
            "metrics": {
                "noise_p95_p50_ratio": round(noise_p95_p50_ratio, 4),
                "noise_block_cv": round(noise_block_cv, 4),
                "global_noise_median": round(p50, 4),
                "face_noise_median": round(face_noise_median, 4)
                if face_noise_median is not None
                else None,
                "bg_noise_median": round(bg_noise_median, 4)
                if bg_noise_median is not None
                else None,
                "face_bg_noise_contrast": round(face_bg_noise_contrast, 4)
                if face_bg_noise_contrast is not None
                else None,
                "face_smoother_than_bg": face_smoother_than_bg,
                "face_chroma_noise": round(face_chroma_noise, 4)
                if face_chroma_noise is not None
                else None,
                "bg_chroma_noise": round(bg_chroma_noise, 4)
                if bg_chroma_noise is not None
                else None,
                "face_bg_chroma_contrast": round(face_bg_chroma_contrast, 4)
                if face_bg_chroma_contrast is not None
                else None,
                "boundary_seam_ratio": round(max_boundary_seam_ratio, 4)
                if max_boundary_seam_ratio is not None
                else None,
            },
            "noise_residual_path": residual_path,
            "noise_heatmap_path": heatmap_path,
        }
