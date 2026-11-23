from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
from PIL import Image

# Canonical pose coordinates for 5 facial landmarks on a 112x112 reference grid
# Order: [left_eye, right_eye, nose, mouth_left, mouth_right]
CANONICAL_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# BGR equivalent of CLIP RGB mean [123, 117, 104]
CLIP_MEAN_BGR = (104, 117, 123)


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Safely read an image with Unicode path support (with Pillow fallback)."""
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
    """Safely save a lossless PNG image with Unicode path support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def _is_valid_face_geometry(
    box: np.ndarray, pts: np.ndarray, min_face_size: int
) -> bool:
    """
    Geometric sanity check to filter out false positives:
    - Enforces minimum face bounding box dimensions for forensic analysis.
    - Validates anatomical ordering of the 5 facial landmarks.
    """
    x1, y1, x2, y2 = box[:4]
    w = float(x2 - x1)
    h = float(y2 - y1)
    if w < min_face_size or h < min_face_size:
        return False

    aspect_ratio = w / max(h, 1.0)
    if aspect_ratio < 0.45 or aspect_ratio > 1.8:
        return False

    left_eye, right_eye, nose, mouth_left, mouth_right = pts
    eye_dx = float(right_eye[0] - left_eye[0])
    if eye_dx <= 0.15 * w:
        return False

    eyes_mid_y = 0.5 * (float(left_eye[1]) + float(right_eye[1]))
    mouth_mid_y = 0.5 * (float(mouth_left[1]) + float(mouth_right[1]))

    if not (eyes_mid_y < float(nose[1]) < mouth_mid_y):
        return False

    return True


def _is_clipped_or_extreme_profile(
    box: np.ndarray, pts: np.ndarray, img_w: int, img_h: int
) -> bool:
    """Flag faces touching frame borders or turned into extreme side profiles."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw = max(x2 - x1, 1.0)

    if x1 < -2.0 or y1 < -2.0 or x2 > (img_w + 2.0) or y2 > (img_h + 2.0):
        return True

    for lx, ly in pts:
        if lx < 0.0 or lx > img_w or ly < 0.0 or ly > img_h:
            return True

    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    eye_dx = abs(float(right_eye[0]) - float(left_eye[0]))
    min_ex = min(float(left_eye[0]), float(right_eye[0]))
    max_ex = max(float(left_eye[0]), float(right_eye[0]))
    nose_x = float(nose[0])

    if eye_dx < 0.24 * bw or nose_x < (min_ex - 0.05 * bw) or nose_x > (max_ex + 0.05 * bw):
        return True

    return False


def _resolve_cv_border(border_mode: str) -> Tuple[int, Tuple[int, int, int]]:
    mode = str(border_mode or "replicate").lower()
    if mode == "constant_black":
        return cv2.BORDER_CONSTANT, (0, 0, 0)
    if mode == "constant_clip_mean":
        return cv2.BORDER_CONSTANT, CLIP_MEAN_BGR
    if mode == "reflect101":
        return cv2.BORDER_REFLECT_101, (0, 0, 0)
    return cv2.BORDER_REPLICATE, (0, 0, 0)


class FaceAligner:
    """
    Module 2.2 & 2.3: Face Localization (RetinaFace / MTCNN) & Single-Pass Multi-Spec Cropping.
    Generates both the canonical display crop and exact model-native crops directly
    from the unedited raw frame without double interpolation.
    """

    def __init__(
        self,
        backend: str = "retinaface",
        output_size: Tuple[int, int] = (256, 256),
        min_face_size: int = 60,
        min_confidence: float = 0.80,
        margin_scale: float = 0.85,
        border_mode: str = "replicate",
        device: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self.margin_scale = margin_scale
        self.border_mode = border_mode
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.backend == "retinaface":
            from uniface import RetinaFace
            from uniface.constants import RetinaFaceWeights

            self.detector = RetinaFace(
                model_name=RetinaFaceWeights.RESNET34,
                confidence_threshold=self.min_confidence,
                nms_threshold=0.4,
            )
        elif self.backend == "mtcnn":
            from facenet_pytorch import MTCNN

            self.detector = MTCNN(
                keep_all=True,
                min_face_size=min_face_size,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose 'retinaface' or 'mtcnn'."
            )

    def _detect(self, img_bgr: np.ndarray) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        detections: List[Tuple[np.ndarray, float, np.ndarray]] = []

        if self.backend == "retinaface":
            faces = self.detector.detect(img_bgr)
            for face in faces:
                prob = float(face.confidence)
                if prob < self.min_confidence:
                    continue
                box = np.asarray(face.bbox, dtype=np.float32)[:4]
                pts = np.asarray(face.landmarks, dtype=np.float32).reshape(5, 2)
                detections.append((box, prob, pts))
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = self.detector.detect(img_rgb, landmarks=True)
            if boxes is not None and probs is not None and landmarks is not None:
                for box, prob, pts in zip(boxes, probs, landmarks):
                    if prob is None or float(prob) < self.min_confidence:
                        continue
                    detections.append(
                        (
                            np.asarray(box, dtype=np.float32)[:4],
                            float(prob),
                            np.asarray(pts, dtype=np.float32).reshape(5, 2),
                        )
                    )

        return detections

    @staticmethod
    def _get_canonical_landmarks(
        output_size: Tuple[int, int], margin_scale: float
    ) -> np.ndarray:
        out_w, out_h = output_size
        norm_pts = CANONICAL_LANDMARKS_112 / 112.0
        centered = (norm_pts - np.array([0.5, 0.5], dtype=np.float32)) * margin_scale
        shifted = centered + np.array([0.5, 0.48], dtype=np.float32)
        shifted[:, 0] *= out_w
        shifted[:, 1] *= out_h
        return shifted.astype(np.float32)

    @staticmethod
    def _estimate_similarity(
        src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if matrix is None:
            matrix = cv2.getAffineTransform(src_pts[[0, 1, 3]], dst_pts[[0, 1, 3]])
        return matrix

    @classmethod
    def _build_crop_matrix(
        cls,
        box: np.ndarray,
        src_landmarks: np.ndarray,
        crop_mode: str,
        output_size: Tuple[int, int],
        margin_scale: float,
        bbox_margin: float,
    ) -> np.ndarray:
        """
        Build a single 2x3 affine transformation matrix directly from raw frame coordinates
        to the target output dimensions, avoiding any intermediate resizing pass.
        """
        out_w, out_h = int(output_size[0]), int(output_size[1])
        src_pts = np.asarray(src_landmarks, dtype=np.float32).reshape(5, 2)
        mode = str(crop_mode or "aligned_square").lower()

        if mode == "raw_bbox":
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            side = max(x2 - x1, y2 - y1, 1.0) * float(bbox_margin)
            x_off = cx - 0.5 * side
            y_off = cy - 0.5 * side
            sx = out_w / side
            sy = out_h / side
            return np.array(
                [[sx, 0.0, -x_off * sx], [0.0, sy, -y_off * sy]],
                dtype=np.float32,
            )

        if mode == "aligned_bbox_margin":
            # DeepfakeBench / GenD protocol composed into one single warp:
            # upright alignment -> aligned bbox enlarged by bbox_margin -> single resize
            m_upright = cls._estimate_similarity(src_pts, CANONICAL_LANDMARKS_112)
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            corners = np.array(
                [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
                dtype=np.float32,
            )
            aligned_corners = corners @ m_upright.T
            ux1, uy1 = aligned_corners.min(axis=0)
            ux2, uy2 = aligned_corners.max(axis=0)
            cx = 0.5 * (float(ux1) + float(ux2))
            cy = 0.5 * (float(uy1) + float(uy2))
            side = max(float(ux2 - ux1), float(uy2 - uy1), 1.0) * float(bbox_margin)

            x_off = cx - 0.5 * side
            y_off = cy - 0.5 * side
            sx = out_w / side
            sy = out_h / side

            s_mat = np.array(
                [[sx, 0.0, -x_off * sx], [0.0, sy, -y_off * sy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            m_upright_3x3 = np.vstack([m_upright, [0.0, 0.0, 1.0]]).astype(np.float32)
            return (s_mat @ m_upright_3x3)[:2, :]

        # Default: "aligned_square"
        dst_pts = cls._get_canonical_landmarks((out_w, out_h), float(margin_scale))
        return cls._estimate_similarity(src_pts, dst_pts)

    @classmethod
    def _warp_with_metrics(
        cls,
        img_bgr: np.ndarray,
        box: np.ndarray,
        src_landmarks: np.ndarray,
        crop_mode: str,
        output_size: Tuple[int, int],
        margin_scale: float,
        bbox_margin: float,
        border_mode: str,
    ) -> Tuple[np.ndarray, float]:
        out_w, out_h = int(output_size[0]), int(output_size[1])
        matrix = cls._build_crop_matrix(
            box=box,
            src_landmarks=src_landmarks,
            crop_mode=crop_mode,
            output_size=(out_w, out_h),
            margin_scale=margin_scale,
            bbox_margin=bbox_margin,
        )
        cv_border, border_val = _resolve_cv_border(border_mode)
        warped = cv2.warpAffine(
            img_bgr,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv_border,
            borderValue=border_val,
        )

        # Measure exact fraction of output pixels sourced from outside the raw frame
        h, w = img_bgr.shape[:2]
        src_mask = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.warpAffine(
            src_mask,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        padded_ratio = round(float(np.mean(warped_mask == 0)), 4)
        return warped, padded_ratio

    def process_frame(
        self,
        frame_path: str,
        aligned_dir: str,
        annotated_dir: Optional[str] = None,
        crop_specs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        src = Path(frame_path)
        img_bgr = _imread_unicode(src)
        if img_bgr is None:
            raise ValueError(f"Unable to read frame image: {src}")

        img_h, img_w = img_bgr.shape[:2]
        detections = self._detect(img_bgr)
        faces_result: List[Dict[str, Any]] = []
        annotated_img = img_bgr.copy() if annotated_dir else None

        # Deduplicate requested model crop specs by spec_id
        unique_specs: Dict[str, Dict[str, Any]] = {}
        for spec in crop_specs or []:
            sid = str(spec.get("spec_id", "")).strip()
            if sid and sid not in unique_specs:
                unique_specs[sid] = spec

        valid_idx = 0
        for box, prob, pts_arr in detections:
            if not _is_valid_face_geometry(box, pts_arr, self.min_face_size):
                continue

            pose_or_clip_warn = _is_clipped_or_extreme_profile(box, pts_arr, img_w, img_h)

            # 1. Standard canonical aligned face crop (for display & default consumers)
            aligned_face, disp_padded_ratio = self._warp_with_metrics(
                img_bgr=img_bgr,
                box=box,
                src_landmarks=pts_arr,
                crop_mode="aligned_square",
                output_size=self.output_size,
                margin_scale=self.margin_scale,
                bbox_margin=1.0,
                border_mode=self.border_mode,
            )

            face_filename = f"{src.stem}_face_{valid_idx:02d}.png"
            aligned_path = Path(aligned_dir) / face_filename
            _imwrite_unicode(aligned_path, aligned_face)

            # 2. Additional model-native single-pass crops (Option B multi-spec export)
            model_crops: Dict[str, Dict[str, Any]] = {}
            for spec_id, spec in unique_specs.items():
                c_mode = str(spec.get("crop_mode", "aligned_square")).lower()
                if c_mode == "raw_frame":
                    model_crops[spec_id] = {
                        "path": str(src.resolve()),
                        "output_size": [img_w, img_h],
                        "crop_mode": "raw_frame",
                        "border_mode": "none",
                        "padded_ratio": 0.0,
                        "unreliable_pose_warning": False,
                    }
                    continue

                out_sz = spec.get("output_size", self.output_size)
                out_tuple = (int(out_sz[0]), int(out_sz[1]))
                m_scale = float(spec.get("margin_scale", self.margin_scale))
                b_margin = float(spec.get("bbox_margin", 1.3))
                b_mode = str(spec.get("border_mode", self.border_mode))

                crop_img, crop_pad_ratio = self._warp_with_metrics(
                    img_bgr=img_bgr,
                    box=box,
                    src_landmarks=pts_arr,
                    crop_mode=c_mode,
                    output_size=out_tuple,
                    margin_scale=m_scale,
                    bbox_margin=b_margin,
                    border_mode=b_mode,
                )
                spec_path = Path(aligned_dir) / spec_id / face_filename
                _imwrite_unicode(spec_path, crop_img)
                model_crops[spec_id] = {
                    "path": str(spec_path.resolve()),
                    "output_size": [out_tuple[0], out_tuple[1]],
                    "crop_mode": c_mode,
                    "border_mode": b_mode,
                    "padded_ratio": crop_pad_ratio,
                    "unreliable_pose_warning": bool(
                        pose_or_clip_warn or crop_pad_ratio > 0.15
                    ),
                }

            x1, y1, x2, y2 = [round(float(v), 2) for v in box]
            face_info = {
                "face_index": valid_idx,
                "detector": self.backend,
                "confidence": round(float(prob), 4),
                "bbox": [x1, y1, x2, y2],
                "landmarks": {
                    "left_eye": [round(float(pts_arr[0][0]), 2), round(float(pts_arr[0][1]), 2)],
                    "right_eye": [round(float(pts_arr[1][0]), 2), round(float(pts_arr[1][1]), 2)],
                    "nose": [round(float(pts_arr[2][0]), 2), round(float(pts_arr[2][1]), 2)],
                    "mouth_left": [round(float(pts_arr[3][0]), 2), round(float(pts_arr[3][1]), 2)],
                    "mouth_right": [round(float(pts_arr[4][0]), 2), round(float(pts_arr[4][1]), 2)],
                },
                "aligned_face_path": str(aligned_path.resolve()),
                "aligned_size": list(self.output_size),
                "padded_ratio": disp_padded_ratio,
                "unreliable_pose_warning": bool(
                    pose_or_clip_warn or disp_padded_ratio > 0.15
                ),
                "model_crops": model_crops,
            }
            faces_result.append(face_info)

            if annotated_img is not None:
                cv2.rectangle(
                    annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                cv2.putText(
                    annotated_img,
                    f"{self.backend.upper()} #{valid_idx} ({prob:.2f})",
                    (int(x1), max(int(y1) - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                for lx, ly in pts_arr:
                    cv2.circle(annotated_img, (int(lx), int(ly)), 3, (0, 0, 255), -1)

            valid_idx += 1

        annotated_path_str: Optional[str] = None
        if annotated_dir and annotated_img is not None:
            ann_path = Path(annotated_dir) / f"{src.stem}_annotated.png"
            _imwrite_unicode(ann_path, annotated_img)
            annotated_path_str = str(ann_path.resolve())

        return {
            "frame_path": str(src.resolve()),
            "image_resolution": [int(img_w), int(img_h)],
            "faces_detected": len(faces_result),
            "annotated_frame_path": annotated_path_str,
            "faces": faces_result,
        }
