from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

    # Nose must lie vertically between eye line and mouth line
    if not (eyes_mid_y < float(nose[1]) < mouth_mid_y):
        return False

    return True


class FaceAligner:
    """
    Module 2.2 & 2.3: Face Localization (RetinaFace / MTCNN) & Affine Transformation
    """

    def __init__(
        self,
        backend: str = "retinaface",
        output_size: Tuple[int, int] = (299, 299),
        min_face_size: int = 60,
        min_confidence: float = 0.80,
        margin_scale: float = 0.85,
        device: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.output_size = output_size
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self.margin_scale = margin_scale
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
            raise ValueError(f"Unsupported backend: {backend}. Choose 'retinaface' or 'mtcnn'.")

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

    def _get_canonical_landmarks(self) -> np.ndarray:
        out_w, out_h = self.output_size
        norm_pts = CANONICAL_LANDMARKS_112 / 112.0
        centered = (norm_pts - np.array([0.5, 0.5], dtype=np.float32)) * self.margin_scale
        shifted = centered + np.array([0.5, 0.48], dtype=np.float32)
        shifted[:, 0] *= out_w
        shifted[:, 1] *= out_h
        return shifted.astype(np.float32)

    def _align_face(self, img_bgr: np.ndarray, src_landmarks: np.ndarray) -> np.ndarray:
        dst_landmarks = self._get_canonical_landmarks()
        src_pts = np.asarray(src_landmarks, dtype=np.float32).reshape(5, 2)

        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_landmarks, method=cv2.LMEDS)
        if matrix is None:
            matrix = cv2.getAffineTransform(src_pts[[0, 1, 3]], dst_landmarks[[0, 1, 3]])

        out_w, out_h = self.output_size
        return cv2.warpAffine(
            img_bgr,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def process_frame(
        self,
        frame_path: str,
        aligned_dir: str,
        annotated_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        src = Path(frame_path)
        img_bgr = _imread_unicode(src)
        if img_bgr is None:
            raise ValueError(f"Unable to read frame image: {src}")

        detections = self._detect(img_bgr)
        faces_result: List[Dict[str, Any]] = []
        annotated_img = img_bgr.copy() if annotated_dir else None

        valid_idx = 0
        for box, prob, pts_arr in detections:
            if not _is_valid_face_geometry(box, pts_arr, self.min_face_size):
                continue

            aligned_face = self._align_face(img_bgr, pts_arr)

            face_filename = f"{src.stem}_face_{valid_idx:02d}.png"
            aligned_path = Path(aligned_dir) / face_filename
            _imwrite_unicode(aligned_path, aligned_face)

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
            "image_resolution": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
            "faces_detected": len(faces_result),
            "annotated_frame_path": annotated_path_str,
            "faces": faces_result,
        }
