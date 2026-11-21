import io
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Safely save a PNG image with Unicode path support on Linux and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def _compute_content_mask(rgb_arr: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """Exclude solid uniform outer margins (e.g., white/black framing) from ELA stats."""
    gray = np.mean(rgb_arr, axis=2)
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


class ELAAnalyzer:
    """
    Module 3.3: Error Level Analysis (ELA) — Native JPEG Specialist.
    Calibrated to avoid false alarms on crisp, unedited real camera JPEGs.
    """

    def __init__(
        self,
        quality: int = 90,
        block_size: int = 16,
        max_amplify: float = 45.0,
    ):
        self.quality = int(np.clip(quality, 50, 99))
        self.block_size = max(4, int(block_size))
        self.max_amplify = float(max_amplify)

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
            raise FileNotFoundError(f"Image file not found for ELA: {src}")

        with Image.open(src) as pil_img:
            original_format = (pil_img.format or src.suffix.lstrip(".")).upper()
            orig_rgb = pil_img.convert("RGB")

            buffer = io.BytesIO()
            orig_rgb.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            with Image.open(buffer) as resaved_img:
                resaved_rgb = resaved_img.convert("RGB")
                resaved_arr = np.asarray(resaved_rgb, dtype=np.float32)

            orig_arr = np.asarray(orig_rgb, dtype=np.float32)

        is_jpeg = original_format in {"JPEG", "JPG"}
        content_mask = _compute_content_mask(orig_arr)

        diff_rgb = np.abs(orig_arr - resaved_arr)
        err_map = np.mean(diff_rgb, axis=2)

        h, w = err_map.shape
        active_err = err_map[content_mask] if np.any(content_mask) else err_map.ravel()
        raw_mean = float(np.mean(active_err))
        raw_std = float(np.std(active_err))
        raw_max = float(np.max(diff_rgb))

        # 1. Local block-grid inconsistency analysis (excluding uniform framing borders)
        bs = min(self.block_size, h, w)
        bh = max(1, h // bs)
        bw = max(1, w // bs)
        cropped = err_map[: bh * bs, : bw * bs]
        c_crop = content_mask[: bh * bs, : bw * bs]

        blocks = cropped.reshape(bh, bs, bw, bs).mean(axis=(1, 3))
        valid_blocks = c_crop.reshape(bh, bs, bw, bs).mean(axis=(1, 3)) >= 0.5
        block_means = blocks[valid_blocks] if np.any(valid_blocks) else blocks.ravel()

        p50 = float(np.percentile(block_means, 50))
        p95 = float(np.percentile(block_means, 95))
        p95_p50_ratio = float(p95 / max(p50, 0.35))
        block_cv = float(np.std(block_means) / max(np.mean(block_means), 0.35))

        hotspot_thresh = raw_mean + 2.5 * raw_std
        hotspot_ratio = float(np.mean(active_err > hotspot_thresh))

        # 2. Compare face ROI vs surrounding background (excluding borders)
        face_region_mean: Optional[float] = None
        bg_region_mean: Optional[float] = None
        face_bg_contrast: Optional[float] = None

        if face_bboxes:
            face_mask = np.zeros((h, w), dtype=bool)
            for box in face_bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    face_mask[y1:y2, x1:x2] = True

            valid_face = face_mask & content_mask
            valid_bg = (~face_mask) & content_mask

            if np.any(valid_face) and np.any(valid_bg):
                face_region_mean = float(np.mean(err_map[valid_face]))
                bg_region_mean = float(np.mean(err_map[valid_bg]))
                face_bg_contrast = float(
                    abs(face_region_mean - bg_region_mean)
                    / max(min(face_region_mean, bg_region_mean), 0.55)
                )

        # 3. Calibrated ELA suspicion score [0.0, 1.0]
        inconsistency_term = float(
            1.0 - np.exp(-0.34 * max(p95_p50_ratio - 1.85, 0.0))
        )
        cv_term = float(np.clip((block_cv - 0.50) / 1.10, 0.0, 1.0))
        hotspot_term = float(np.clip((hotspot_ratio - 0.02) / 0.07, 0.0, 1.0))

        subscores = [inconsistency_term, cv_term, hotspot_term]
        if face_bg_contrast is not None:
            face_term = float(
                1.0 - np.exp(-0.95 * max(face_bg_contrast - 0.28, 0.0))
            )
            subscores.append(face_term)
            weighted_avg = (
                0.35 * face_term
                + 0.30 * inconsistency_term
                + 0.20 * cv_term
                + 0.15 * hotspot_term
            )
        else:
            weighted_avg = (
                0.45 * inconsistency_term
                + 0.35 * cv_term
                + 0.20 * hotspot_term
            )

        ela_score = round(
            float(np.clip(0.35 * max(subscores) + 0.65 * weighted_avg, 0.0, 1.0)),
            4,
        )

        # 4. Generate ELA visualizations
        ela_rgb_path: Optional[str] = None
        ela_heatmap_path: Optional[str] = None

        if output_dir:
            out_path = Path(output_dir)
            tag = prefix or src.stem

            p995_rgb = float(np.percentile(diff_rgb, 99.5))
            ref_rgb = p995_rgb if p995_rgb > 1e-3 else raw_max
            rgb_scale = min(255.0 / max(ref_rgb, 1e-6), self.max_amplify)

            norm_rgb = np.clip((diff_rgb * rgb_scale) / 255.0, 0.0, 1.0)
            amplified_rgb = np.clip(
                np.power(norm_rgb, 0.80) * 255.0, 0, 255
            ).astype(np.uint8)
            amplified_bgr = cv2.cvtColor(amplified_rgb, cv2.COLOR_RGB2BGR)

            p995_err = float(np.percentile(err_map, 99.5))
            ref_err = p995_err if p995_err > 1e-3 else float(np.max(err_map))
            err_scale = min(255.0 / max(ref_err, 1e-6), self.max_amplify)

            norm_err_01 = np.clip((err_map * err_scale) / 255.0, 0.0, 1.0)
            norm_err = np.clip(
                np.power(norm_err_01, 0.75) * 255.0, 0, 255
            ).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(norm_err, cv2.COLORMAP_INFERNO)

            rgb_file = out_path / f"{tag}_ela_raw.png"
            heat_file = out_path / f"{tag}_ela_heatmap.png"
            _imwrite_unicode(rgb_file, amplified_bgr)
            _imwrite_unicode(heat_file, heatmap_bgr)

            ela_rgb_path = str(rgb_file.resolve())
            ela_heatmap_path = str(heat_file.resolve())

        note = (
            "Original format is JPEG — well suited for double-compression ELA and included in Module 3 scoring."
            if is_jpeg
            else f"Input format is {original_format} (e.g. extracted video PNG frame); ELA is shown for reference only and excluded from Module 3 scoring."
        )

        return {
            "source_path": str(src.resolve()),
            "original_format": original_format,
            "is_jpeg": is_jpeg,
            "ela_quality": self.quality,
            "ela_score": ela_score,
            "suspicion_level": self._compute_suspicion_level(ela_score),
            "metrics": {
                "mean_error": round(raw_mean, 4),
                "std_error": round(raw_std, 4),
                "max_error": round(raw_max, 4),
                "block_p95_p50_ratio": round(p95_p50_ratio, 4),
                "block_cv": round(block_cv, 4),
                "hotspot_ratio": round(hotspot_ratio, 5),
                "face_region_mean": round(face_region_mean, 4)
                if face_region_mean is not None
                else None,
                "bg_region_mean": round(bg_region_mean, 4)
                if bg_region_mean is not None
                else None,
                "face_bg_contrast": round(face_bg_contrast, 4)
                if face_bg_contrast is not None
                else None,
            },
            "ela_raw_path": ela_rgb_path,
            "ela_heatmap_path": ela_heatmap_path,
            "note": note,
        }
