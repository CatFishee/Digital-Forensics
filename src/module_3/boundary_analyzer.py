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


def _check_face_geometry_pose(
    face: Dict[str, Any], img_w: int, img_h: int
) -> Tuple[bool, bool]:
    """
    Distinguish between:
      - is_edge_clipped: Face physically touches/exceeds the image border.
      - is_asymmetric_splice: Severe bilateral or landmark displacement occurring
        on a predominantly frontal face (an indicator of a spliced / chimera face).
    """
    bbox = face.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return False, False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    # Edge clipping check
    is_edge_clipped = (
        x1 < -2.0 or y1 < -2.0 or x2 > (img_w + 2.0) or y2 > (img_h + 2.0)
    )

    lm = face.get("landmarks") or {}
    for pt in lm.values():
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            if pt[0] < 0 or pt[0] > img_w or pt[1] < 0 or pt[1] > img_h:
                is_edge_clipped = True

    # Bilateral landmark symmetry check on frontal faces
    is_asymmetric_splice = False
    le = lm.get("left_eye")
    re = lm.get("right_eye")
    nose = lm.get("nose")
    ml = lm.get("mouth_left")
    mr = lm.get("mouth_right")

    if le and re and nose and ml and mr and not is_edge_clipped:
        eye_dx = abs(float(re[0]) - float(le[0]))
        min_ex = min(float(le[0]), float(re[0]))
        max_ex = max(float(le[0]), float(re[0]))
        nose_x = float(nose[0])

        # If both eyes and mouth span a normal frontal face width (>0.32 bw)
        # but nose is severely off-center or eye level is heavily warped:
        mouth_dx = abs(float(mr[0]) - float(ml[0]))
        if eye_dx > 0.30 * bw and mouth_dx > 0.25 * bw:
            nose_center_offset = abs(nose_x - 0.5 * (float(le[0]) + float(re[0])))
            if nose_center_offset > 0.22 * eye_dx:
                is_asymmetric_splice = True
        elif eye_dx < 0.24 * bw or nose_x < (min_ex - 0.05 * bw) or nose_x > (max_ex + 0.05 * bw):
            is_edge_clipped = True

    return is_edge_clipped, is_asymmetric_splice


class BoundaryAnalyzer:
    """
    Module 3.2: Face-Boundary Blending & Internal Splice Seam Analysis.
    Evaluates:
      1. Outer jawline/cheek perimeter blending halos and skin-context color bleed.
      2. Internal face splice seams & inpainting edges across the nose, eyes, and midface.
      3. Bilateral facial chrominance & gradient symmetry (left vs. right face half).
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
                    "internal_seam_ratio": None,
                    "bilateral_chroma_diff": None,
                    "bilateral_gradient_ratio": None,
                    "faces_evaluated": 0,
                    "unreliable_face_warning": False,
                    "anatomical_asymmetry_detected": False,
                },
                "boundary_overlay_path": None,
                "boundary_heatmap_path": None,
            }

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1] - 128.0
        b_chan = lab[:, :, 2] - 128.0

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap_lum = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        lap_chroma = 0.5 * (
            np.abs(cv2.Laplacian(a_chan, cv2.CV_32F, ksize=3))
            + np.abs(cv2.Laplacian(b_chan, cv2.CV_32F, ksize=3))
        )
        local_anomaly_map = lap_chroma * 1.8 + 0.35 * lap_lum

        yy, xx = np.indices((h, w), dtype=np.float32)

        seam_halo_vals: List[float] = []
        chroma_diff_vals: List[float] = []
        chroma_bleed_vals: List[float] = []
        sharp_ratio_vals: List[float] = []
        internal_seam_vals: List[float] = []
        bilateral_chroma_vals: List[float] = []
        bilateral_grad_vals: List[float] = []

        any_edge_clipped = False
        any_asymmetric_splice = False

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

            is_clipped, is_asym = _check_face_geometry_pose(face, w, h)
            if is_clipped:
                any_edge_clipped = True
            if is_asym:
                any_asymmetric_splice = True

            cx, cy, rx, ry, angle = self._fit_face_oval(face, w, h)
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)

            dx = xx - cx
            dy = yy - cy
            xr = dx * cos_a - dy * sin_a
            yr = dx * sin_a + dy * cos_a

            r_oval = np.sqrt((xr / max(rx, 1.0)) ** 2 + (yr / max(ry, 1.0)) ** 2)
            lower_face = yr >= (-0.20 * ry)

            # 1. Outer perimeter masks (jawline and cheek blending)
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

            # 2. Internal face anomaly analysis (closing the Heatmap-to-Metric gap)
            # Create anatomical exclusion masks for genuine facial feature points
            feature_mask = np.zeros((h, w), dtype=bool)
            lm = face.get("landmarks") or {}
            for k in ("left_eye", "right_eye", "mouth_left", "mouth_right", "nose"):
                pt = lm.get(k)
                if pt and len(pt) >= 2:
                    px_c, py_c = int(round(pt[0])), int(round(pt[1]))
                    rad_ex = int(round(0.12 * bw_box)) if "mouth" in k else int(round(0.09 * bw_box))
                    y1_ex, y2_ex = max(0, py_c - rad_ex), min(h, py_c + rad_ex)
                    x1_ex, x2_ex = max(0, px_c - rad_ex), min(w, px_c + rad_ex)
                    feature_mask[y1_ex:y2_ex, x1_ex:x2_ex] = True

            # Internal face skin zone (inside cheek/jaw bounds, excluding natural feature centers)
            internal_skin_mask = (r_oval <= 0.82) & (~feature_mask)
            if np.sum(internal_skin_mask) > 100:
                face_skin_anomalies = local_anomaly_map[internal_skin_mask]
                p95_internal = float(np.percentile(face_skin_anomalies, 95))
                p50_internal = float(np.percentile(face_skin_anomalies, 50))
                internal_seam_ratio = p95_internal / max(p50_internal, 1.2)
                internal_seam_vals.append(internal_seam_ratio)
            else:
                internal_seam_vals.append(1.0)

            # 3. Bilateral facial symmetry (left half vs. right half of face)
            left_face_mask = (r_oval <= 0.80) & (xr < -0.06 * rx) & (~feature_mask)
            right_face_mask = (r_oval <= 0.80) & (xr > 0.06 * rx) & (~feature_mask)
            if np.sum(left_face_mask) > 80 and np.sum(right_face_mask) > 80:
                l_a = float(np.median(a_chan[left_face_mask]))
                l_b = float(np.median(b_chan[left_face_mask]))
                r_a = float(np.median(a_chan[right_face_mask]))
                r_b = float(np.median(b_chan[right_face_mask]))
                bilateral_delta = math.sqrt((l_a - r_a) ** 2 + (l_b - r_b) ** 2)
                bilateral_chroma_vals.append(bilateral_delta)

                l_grad = float(np.percentile(local_anomaly_map[left_face_mask], 75))
                r_grad = float(np.percentile(local_anomaly_map[right_face_mask], 75))
                grad_rat = max(l_grad, r_grad) / max(min(l_grad, r_grad), 1.0)
                bilateral_grad_vals.append(grad_rat)
            else:
                bilateral_chroma_vals.append(0.0)
                bilateral_grad_vals.append(1.0)

            # Accumulate heatmap across entire inspected face zone
            face_zone = (r_oval <= 1.44)
            combined_seam_energy[face_zone] = np.maximum(
                combined_seam_energy[face_zone], local_anomaly_map[face_zone]
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
                    "internal_seam_ratio": None,
                    "bilateral_chroma_diff": None,
                    "bilateral_gradient_ratio": None,
                    "faces_evaluated": 0,
                    "unreliable_face_warning": any_edge_clipped,
                    "anatomical_asymmetry_detected": any_asymmetric_splice,
                },
                "boundary_overlay_path": None,
                "boundary_heatmap_path": None,
            }

        max_seam_halo = float(max(seam_halo_vals))
        max_chroma_diff = float(max(chroma_diff_vals))
        max_chroma_bleed = float(max(chroma_bleed_vals))
        max_sharp_mismatch = float(max(sharp_ratio_vals))
        max_internal_seam = float(max(internal_seam_vals))
        max_bilateral_chroma = float(max(bilateral_chroma_vals))
        max_bilateral_grad = float(max(bilateral_grad_vals))

        # Calibrated subscores
        halo_term = float(np.clip((max_seam_halo - 1.90) / 1.40, 0.0, 1.0))
        bleed_term = float(np.clip((max_chroma_bleed - 1.75) / 1.25, 0.0, 1.0))
        chroma_term = float(np.clip((max_chroma_diff - 11.0) / 16.0, 0.0, 1.0))
        sharp_term = float(np.clip((max_sharp_mismatch - 2.30) / 2.20, 0.0, 1.0))

        # Internal splice terms (catches nose/eye inpainting seams and skin tone split)
        internal_seam_term = float(np.clip((max_internal_seam - 2.40) / 1.80, 0.0, 1.0))
        bilateral_chroma_term = float(np.clip((max_bilateral_chroma - 7.50) / 9.0, 0.0, 1.0))
        bilateral_grad_term = float(np.clip((max_bilateral_grad - 1.80) / 1.50, 0.0, 1.0))
        asym_term = 0.50 if any_asymmetric_splice else 0.0

        perimeter_subscore = float(
            np.clip(
                0.55 * max([halo_term, bleed_term, chroma_term, sharp_term])
                + 0.45 * float(np.mean([halo_term, bleed_term, chroma_term, sharp_term])),
                0.0,
                1.0,
            )
        )
        internal_subscore = float(
            np.clip(
                0.50 * max([internal_seam_term, bilateral_chroma_term, bilateral_grad_term])
                + 0.35 * float(np.mean([internal_seam_term, bilateral_chroma_term, bilateral_grad_term]))
                + 0.15 * asym_term,
                0.0,
                1.0,
            )
        )

        boundary_score = round(
            float(
                np.clip(
                    0.60 * max(perimeter_subscore, internal_subscore)
                    + 0.40 * (0.5 * (perimeter_subscore + internal_subscore)),
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
                "internal_seam_ratio": round(max_internal_seam, 4),
                "bilateral_chroma_diff": round(max_bilateral_chroma, 4),
                "bilateral_gradient_ratio": round(max_bilateral_grad, 4),
                "faces_evaluated": len(seam_halo_vals),
                "unreliable_face_warning": any_edge_clipped,
                "anatomical_asymmetry_detected": any_asymmetric_splice,
            },
            "boundary_overlay_path": overlay_path,
            "boundary_heatmap_path": heatmap_path,
        }
