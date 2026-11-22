import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


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


def _is_clipped_or_extreme_profile(
    face: Dict[str, Any], img_w: int, img_h: int
) -> bool:
    """
    Detect whether a face is partially cut off at the frame border or turned into
    an extreme side profile (used to attach an unreliability warning for users).
    """
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


class BoundaryAnalyzer:
    """
    Module 3.2: Face-Boundary Blending & Chrominance Mismatch Analysis.
    Evaluates all detected faces and flags extreme side profiles / partially covered
    faces so the UI can warn the user when results may be unreliable.
    """

    @staticmethod
    def _compute_suspicion_level(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    @staticmethod
    def _fit_face_oval(
        face: Dict[str, Any], img_w: int, img_h: int
    ) -> Tuple[float, float, float, float, float]:
        """Return (cx, cy, rx, ry, angle_rad) for an anatomical face oval."""
        bbox = face.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)

        landmarks = face.get("landmarks") or {}
        le = landmarks.get("left_eye")
        re = landmarks.get("right_eye")
        ml = landmarks.get("mouth_left")
        mr = landmarks.get("mouth_right")

        if le and re and ml and mr:
            eye_mid_x = 0.5 * (float(le[0]) + float(re[0]))
            eye_mid_y = 0.5 * (float(le[1]) + float(re[1]))
            mouth_mid_x = 0.5 * (float(ml[0]) + float(mr[0]))
            mouth_mid_y = 0.5 * (float(ml[1]) + float(mr[1]))

            angle = math.atan2(float(re[1]) - float(le[1]), float(re[0]) - float(le[0]))
            cx = 0.5 * (eye_mid_x + mouth_mid_x)
            cy = 0.45 * eye_mid_y + 0.55 * mouth_mid_y
            rx = float(np.clip(0.44 * bw, 12.0, 0.55 * img_w))
            ry = float(np.clip(0.48 * bh, 14.0, 0.55 * img_h))
        else:
            angle = 0.0
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            rx = max(0.44 * bw, 12.0)
            ry = max(0.48 * bh, 14.0)

        return cx, cy, rx, ry, angle

    def analyze(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        faces: Optional[Sequence[Dict[str, Any]]] = None,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image file not found for Boundary Analysis: {src}")

        img_bgr = _imread_bgr_unicode(src)
        if img_bgr is None:
            raise ValueError(f"Unable to read image for Boundary Analysis: {src}")

        h, w = img_bgr.shape[:2]
        faces_list = list(faces or [])

        if not faces_list:
            return {
                "source_path": str(src.resolve()),
                "faces_analyzed": 0,
                "boundary_score": 0.0,
                "suspicion_level": "low",
                "metrics": {
                    "seam_halo_ratio": None,
                    "skin_context_chroma_diff": None,
                    "seam_chroma_bleed": None,
                    "inner_outer_sharpness_ratio": None,
                    "faces_evaluated": 0,
                    "unreliable_face_warning": False,
                },
                "boundary_overlay_path": None,
                "boundary_heatmap_path": None,
            }

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_chan = lab[:, :, 1] - 128.0
        b_chan = lab[:, :, 2] - 128.0

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap_lum = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        lap_chroma = 0.5 * (
            np.abs(cv2.Laplacian(a_chan, cv2.CV_32F, ksize=3))
            + np.abs(cv2.Laplacian(b_chan, cv2.CV_32F, ksize=3))
        )

        yy, xx = np.indices((h, w), dtype=np.float32)

        seam_halo_vals: List[float] = []
        chroma_diff_vals: List[float] = []
        chroma_bleed_vals: List[float] = []
        sharp_ratio_vals: List[float] = []
        any_unreliable = False

        overlay_img = img_bgr.copy()
        combined_seam_energy = np.zeros((h, w), dtype=np.float32)

        for face in faces_list:
            bbox = face.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            bw_box = float(bbox[2]) - float(bbox[0])
            bh_box = float(bbox[3]) - float(bbox[1])
            if bw_box < 32 or bh_box < 32:
                continue

            if _is_clipped_or_extreme_profile(face, w, h):
                any_unreliable = True

            cx, cy, rx, ry, angle = self._fit_face_oval(face, w, h)
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)

            dx = xx - cx
            dy = yy - cy
            xr = dx * cos_a - dy * sin_a
            yr = dx * sin_a + dy * cos_a

            r_oval = np.sqrt((xr / max(rx, 1.0)) ** 2 + (yr / max(ry, 1.0)) ** 2)
            lower_face = yr >= (-0.20 * ry)

            inner_skin_mask = (r_oval >= 0.25) & (r_oval <= 0.72) & lower_face
            seam_mask = (r_oval >= 0.88) & (r_oval <= 1.12) & lower_face
            outer_mask = (r_oval >= 1.16) & (r_oval <= 1.44) & lower_face

            if not (np.any(inner_skin_mask) and np.any(seam_mask) and np.any(outer_mask)):
                continue

            seam_lap = float(np.median(lap_lum[seam_mask]))
            ref_lap = 0.5 * (
                float(np.median(lap_lum[inner_skin_mask]))
                + float(np.median(lap_lum[outer_mask]))
            )
            seam_halo_vals.append(seam_lap / max(ref_lap, 2.0))

            seam_c = float(np.percentile(lap_chroma[seam_mask], 65))
            ref_c = 0.5 * (
                float(np.percentile(lap_chroma[inner_skin_mask], 65))
                + float(np.percentile(lap_chroma[outer_mask], 65))
            )
            chroma_bleed_vals.append(seam_c / max(ref_c, 1.2))

            in_a = float(np.median(a_chan[inner_skin_mask]))
            in_b = float(np.median(b_chan[inner_skin_mask]))
            out_a = a_chan[outer_mask]
            out_b = b_chan[outer_mask]
            delta_ab = np.sqrt((out_a - in_a) ** 2 + (out_b - in_b) ** 2)
            chroma_diff_vals.append(float(np.percentile(delta_ab, 35)))

            inner_sharp = float(np.percentile(lap_lum[inner_skin_mask], 80))
            outer_sharp = float(np.percentile(lap_lum[outer_mask], 80))
            sharp_ratio = max(inner_sharp, 2.5) / max(outer_sharp, 2.5)
            inv_ratio = max(sharp_ratio, 1.0 / max(sharp_ratio, 1e-3))
            sharp_ratio_vals.append(inv_ratio)

            face_zone = (r_oval <= 1.44) & lower_face
            local_anomaly = lap_chroma * 1.8 + 0.35 * lap_lum
            combined_seam_energy[face_zone] = np.maximum(
                combined_seam_energy[face_zone], local_anomaly[face_zone]
            )

            center_pt = (int(round(cx)), int(round(cy)))
            axes_inner = (max(1, int(round(rx * 0.88))), max(1, int(round(ry * 0.88))))
            axes_outer = (max(1, int(round(rx * 1.12))), max(1, int(round(ry * 1.12))))
            deg = float(math.degrees(angle))
            cv2.ellipse(
                overlay_img, center_pt, axes_inner, deg, -15, 195, (0, 255, 255), 2
            )
            cv2.ellipse(
                overlay_img, center_pt, axes_outer, deg, -15, 195, (255, 140, 0), 2
            )

        if not seam_halo_vals:
            return {
                "source_path": str(src.resolve()),
                "faces_analyzed": len(faces_list),
                "boundary_score": 0.0,
                "suspicion_level": "low",
                "metrics": {
                    "seam_halo_ratio": None,
                    "skin_context_chroma_diff": None,
                    "seam_chroma_bleed": None,
                    "inner_outer_sharpness_ratio": None,
                    "faces_evaluated": 0,
                    "unreliable_face_warning": any_unreliable,
                },
                "boundary_overlay_path": None,
                "boundary_heatmap_path": None,
            }

        max_seam_halo = float(max(seam_halo_vals))
        max_chroma_diff = float(max(chroma_diff_vals))
        max_chroma_bleed = float(max(chroma_bleed_vals))
        max_sharp_mismatch = float(max(sharp_ratio_vals))

        halo_term = float(np.clip((max_seam_halo - 1.90) / 1.40, 0.0, 1.0))
        bleed_term = float(np.clip((max_chroma_bleed - 1.75) / 1.25, 0.0, 1.0))
        chroma_term = float(np.clip((max_chroma_diff - 11.0) / 16.0, 0.0, 1.0))
        sharp_term = float(np.clip((max_sharp_mismatch - 2.30) / 2.20, 0.0, 1.0))

        subscores = [halo_term, bleed_term, chroma_term, sharp_term]
        boundary_score = round(
            float(
                np.clip(
                    0.55 * max(subscores) + 0.45 * float(np.mean(subscores)),
                    0.0,
                    1.0,
                )
            ),
            4,
        )

        overlay_path: Optional[str] = None
        heatmap_path: Optional[str] = None
        if output_dir:
            out_path = Path(output_dir)
            tag = prefix or src.stem

            nz = combined_seam_energy[combined_seam_energy > 0]
            ref = float(np.percentile(nz, 98.0)) if nz.size > 0 else 1.0
            norm = np.clip(combined_seam_energy / max(ref, 1.0), 0.0, 1.0)
            norm_u8 = np.clip(np.power(norm, 0.75) * 255.0, 0, 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
            blended_heat = cv2.addWeighted(img_bgr, 0.40, heat_bgr, 0.60, 0.0)

            ov_file = out_path / f"{tag}_boundary_zones.png"
            hm_file = out_path / f"{tag}_boundary_heatmap.png"
            _imwrite_unicode(ov_file, overlay_img)
            _imwrite_unicode(hm_file, blended_heat)
            overlay_path = str(ov_file.resolve())
            heatmap_path = str(hm_file.resolve())

        return {
            "source_path": str(src.resolve()),
            "faces_analyzed": len(faces_list),
            "boundary_score": boundary_score,
            "suspicion_level": self._compute_suspicion_level(boundary_score),
            "metrics": {
                "seam_halo_ratio": round(max_seam_halo, 4),
                "skin_context_chroma_diff": round(max_chroma_diff, 4),
                "seam_chroma_bleed": round(max_chroma_bleed, 4),
                "inner_outer_sharpness_ratio": round(max_sharp_mismatch, 4),
                "faces_evaluated": len(seam_halo_vals),
                "unreliable_face_warning": any_unreliable,
            },
            "boundary_overlay_path": overlay_path,
            "boundary_heatmap_path": heatmap_path,
        }
