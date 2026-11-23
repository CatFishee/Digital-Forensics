from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .base import (
    BaseModule4Detector,
    CropSpec,
    compute_suspicion_level,
    resolve_torch_device,
)


def _reshape_transform_vit(tensor: torch.Tensor) -> torch.Tensor:
    """Biến đổi chuỗi token 1D của ViT thành tensor không gian 2D cho Grad-CAM."""
    if tensor.dim() == 3:
        num_patches = tensor.size(1) - 1
        side = int(round(num_patches ** 0.5))
        if side * side == num_patches:
            result = tensor[:, 1:, :].reshape(tensor.size(0), side, side, tensor.size(2))
            return result.permute(0, 3, 1, 2)
    return tensor


class _HFLogitsWrapper(nn.Module):
    """Bọc ViTForImageClassification để trả về Tensor logits thuần cho pytorch_grad_cam."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf_model(pixel_values=x).logits


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


class CommunityForensicsDetector(BaseModule4Detector):
    """
    Module 4 Plugin: Community Forensics ViT-S/16 (CVPR 2025)
    - Phân tích toàn khung hình (raw_frame: shortest_edge 440 -> center_crop 384x384).
    - Hỗ trợ Grad-CAM trên cả transformers >= 5.x (vit.layers) và 4.x (vit.encoder.layer).
    """

    detector_id: str = "commfor"
    display_name: str = "Community Forensics ViT-S/16 (CVPR 2025)"
    paradigm: str = "AI-Generated Full-Frame Forensics"
    input_size: int = 384
    input_source: str = "raw_frame"
    requires_face: bool = False
    crop_spec: Optional[CropSpec] = CropSpec(
        spec_id="commfor_raw_frame",
        crop_mode="raw_frame",
        output_size=(384, 384),
        border_mode="none",
    )

    DEFAULT_HF_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        hf_model_name: str = DEFAULT_HF_MODEL,
    ) -> None:
        super().__init__(weights_path=weights_path)
        self.hf_model_name = hf_model_name
        self.model: Optional[nn.Module] = None
        self.processor = None
        self._fallback_transform = None
        self.cam = None
        self._loaded_weights_path: str = hf_model_name

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def _get_norm_stats(self) -> Tuple[List[float], List[float]]:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        if self.processor is not None:
            mean = list(getattr(self.processor, "image_mean", mean))
            std = list(getattr(self.processor, "image_std", std))
        return mean, std

    def _prepare_native_tensor(self, pil_img: Image.Image, crop_size: int = 384) -> torch.Tensor:
        mean, std = self._get_norm_stats()
        tf = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return tf(pil_img).unsqueeze(0).to(self.device)

    def _build_fallback_transform(self, source_ref: str) -> transforms.Compose:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        resize_short = 440
        crop_size = 384

        try:
            cfg_file = Path(source_ref) / "preprocessor_config.json"
            if not cfg_file.exists():
                from huggingface_hub import hf_hub_download

                cfg_file = Path(hf_hub_download(source_ref, "preprocessor_config.json"))
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            mean = cfg.get("image_mean", mean)
            std = cfg.get("image_std", std)
            if isinstance(cfg.get("size"), dict):
                resize_short = int(cfg["size"].get("shortest_edge", resize_short))
            if isinstance(cfg.get("crop_size"), dict):
                crop_size = int(cfg["crop_size"].get("height", crop_size))
        except Exception:
            pass

        return transforms.Compose(
            [
                transforms.Resize(
                    resize_short, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def load(self, device: str = "auto", auto_download: bool = True) -> None:
        if self._is_loaded and self.model is not None:
            return

        self.device = resolve_torch_device(device)
        weights_ref = self.resolve_weights_path(auto_download=auto_download)
        weights_p = Path(weights_ref)

        hf_source = str(weights_p) if weights_p.is_dir() else self.hf_model_name
        try:
            self.processor = AutoImageProcessor.from_pretrained(hf_source)
            self._fallback_transform = None
        except Exception:
            self.processor = None
            self._fallback_transform = self._build_fallback_transform(hf_source)

        self.model = AutoModelForImageClassification.from_pretrained(hf_source)

        if weights_p.is_file():
            checkpoint = torch.load(str(weights_p), map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            total_keys = max(len(self.model.state_dict()), 1)
            if len(incompatible.missing_keys) > 0.25 * total_keys:
                raise RuntimeError(
                    f"Checkpoint architecture mismatch! Missing {len(incompatible.missing_keys)}/{total_keys} keys."
                )

        self.model.to(self.device)
        self.model.eval()
        self._loaded_weights_path = str(weights_p.resolve())

        from pytorch_grad_cam import GradCAM

        try:
            vit_backbone = getattr(self.model, "vit", self.model)
            if hasattr(vit_backbone, "layers"):
                # transformers >= 5.x
                last_block = vit_backbone.layers[-1]
            elif hasattr(vit_backbone, "encoder") and hasattr(vit_backbone.encoder, "layer"):
                # transformers 4.x
                last_block = vit_backbone.encoder.layer[-1]
            elif hasattr(vit_backbone, "blocks"):
                last_block = vit_backbone.blocks[-1]
            else:
                raise AttributeError("Unsupported ViT architecture: cannot locate transformer blocks.")

            target_layer = getattr(
                last_block, "layernorm_before", getattr(last_block, "norm1", last_block)
            )
            self.cam = GradCAM(
                model=_HFLogitsWrapper(self.model),
                target_layers=[target_layer],
                reshape_transform=_reshape_transform_vit,
            )
        except Exception as exc:
            print(f"[!] Warning: Grad-CAM initialization failed for commfor: {exc}")
            self.cam = None

        self._is_loaded = True

    def unload(self) -> None:
        if self.cam is not None:
            del self.cam
            self.cam = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._fallback_transform = None
        super().unload()

    def _prepare_input_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        if self.processor is not None:
            inputs = self.processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
        assert self._fallback_transform is not None
        return self._fallback_transform(pil_img).unsqueeze(0).to(self.device)

    def _infer_probabilities(
        self, logits: torch.Tensor
    ) -> Tuple[float, float, int, Dict[str, float]]:
        num_classes = int(logits.shape[-1])
        if num_classes == 1:
            raw_logit = float(logits[0, 0].item())
            prob_fake = float(torch.sigmoid(logits[0, 0]).item())
            prob_real = 1.0 - prob_fake
            return (
                prob_fake,
                prob_real,
                0,
                {"real": round(-raw_logit, 4), "fake": round(raw_logit, 4)},
            )

        probs = torch.softmax(logits, dim=-1)[0]
        id2label = getattr(self.model.config, "id2label", {}) or {}
        label2id = {str(v).lower(): int(k) for k, v in id2label.items()}
        fake_idx = label2id.get("fake", label2id.get("ai", 1))
        real_idx = label2id.get("real", label2id.get("human", 0 if fake_idx == 1 else 1))

        prob_fake = float(probs[fake_idx].item())
        prob_real = float(probs[real_idx].item())
        return (
            prob_fake,
            prob_real,
            fake_idx,
            {
                "real": round(float(logits[0, real_idx].item()), 4),
                "fake": round(float(logits[0, fake_idx].item()), 4),
            },
        )

    def _build_visual_crop(self, pil_img: Image.Image, size: int) -> Image.Image:
        resize_short = int(round(size * 440 / 384))
        return transforms.CenterCrop(size)(
            transforms.Resize(
                resize_short, interpolation=transforms.InterpolationMode.BICUBIC
            )(pil_img)
        )

    def analyze_case(
        self,
        m2_manifest: Dict[str, Any],
        output_dir: str,
        batch_size: int = 4,
        enable_gradcam: bool = True,
    ) -> Dict[str, Any]:
        if not self._is_loaded or self.model is None:
            raise RuntimeError("CommunityForensicsDetector must be loaded before analyze_case().")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_detail = m2_manifest.get("frames_detail", []) or []

        frame_results: List[Dict[str, Any]] = []
        eval_scores: List[float] = []

        for idx, frame_rec in enumerate(frames_detail):
            frame_idx = int(frame_rec.get("frame_display_index", idx + 1))
            frame_path = frame_rec.get("frame_path", "")

            if not frame_path or not Path(frame_path).exists():
                frame_results.append(
                    {
                        "frame_index": frame_idx,
                        "frame_path": frame_path,
                        "evaluated": False,
                        "fake_probability": None,
                        "suspicion_level": "low",
                        "peak_face_id": None,
                        "faces": [],
                    }
                )
                continue

            pil_img = Image.open(frame_path).convert("RGB")
            t_default = self._prepare_input_tensor(pil_img)
            cam_h, cam_w = int(t_default.shape[2]), int(t_default.shape[3])

            use_dual_scale = min(pil_img.size) >= 440 and cam_h == cam_w
            if use_dual_scale:
                t_native = self._prepare_native_tensor(pil_img, crop_size=cam_h)
                batch_in = torch.cat([t_default, t_native], dim=0)
                with torch.no_grad():
                    batch_logits = self.model(pixel_values=batch_in).logits
                p_def, _, idx_def, logits_def = self._infer_probabilities(batch_logits[0:1])
                p_nat, _, idx_nat, logits_nat = self._infer_probabilities(batch_logits[1:2])

                prob_fake = 0.75 * max(p_def, p_nat) + 0.25 * min(p_def, p_nat)
                prob_real = 1.0 - prob_fake
                if p_nat > p_def:
                    input_tensor = t_native
                    cam_target_idx = idx_nat
                    logits_dict = logits_nat
                    vis_pil = transforms.CenterCrop(cam_h)(pil_img)
                else:
                    input_tensor = t_default
                    cam_target_idx = idx_def
                    logits_dict = logits_def
                    vis_pil = self._build_visual_crop(pil_img, cam_h)
            else:
                input_tensor = t_default
                with torch.no_grad():
                    logits = self.model(pixel_values=input_tensor).logits
                prob_fake, prob_real, cam_target_idx, logits_dict = self._infer_probabilities(logits)
                vis_pil = (
                    self._build_visual_crop(pil_img, cam_h)
                    if cam_h == cam_w
                    else pil_img.resize((cam_w, cam_h), Image.Resampling.BICUBIC)
                )

            sev = compute_suspicion_level(prob_fake)
            pred_label = "FAKE" if prob_fake >= 0.50 else "REAL"
            eval_scores.append(prob_fake)

            vis_rgb = np.array(vis_pil, dtype=np.uint8)
            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

            frame_stem = Path(frame_path).stem
            crop_file = out_dir / f"{frame_stem}_commfor_crop.png"
            crop_path_str = str(crop_file.resolve()) if _imwrite_unicode(crop_file, vis_bgr) else frame_path

            gradcam_path_str: Optional[str] = None
            xai_metrics = {
                "max_activation": 0.0,
                "mean_activation": 0.0,
                "high_activation_area_ratio": 0.0,
            }

            if enable_gradcam and self.cam is not None:
                try:
                    from pytorch_grad_cam.utils.image import show_cam_on_image
                    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

                    targets = [ClassifierOutputTarget(cam_target_idx)]
                    with torch.enable_grad():
                        grayscale_cam = self.cam(
                            input_tensor=input_tensor, targets=targets
                        )[0, :]

                    img_float = vis_rgb.astype(np.float32) / 255.0
                    overlay_rgb = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
                    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    cam_file = out_dir / f"{frame_stem}_commfor_gradcam.png"
                    if _imwrite_unicode(cam_file, overlay_bgr):
                        gradcam_path_str = str(cam_file.resolve())

                    xai_metrics = {
                        "max_activation": round(float(np.max(grayscale_cam)), 4),
                        "mean_activation": round(float(np.mean(grayscale_cam)), 4),
                        "high_activation_area_ratio": round(
                            float(np.mean(grayscale_cam > 0.6)), 4
                        ),
                    }
                except Exception as exc:
                    print(f"[!] Error generating Grad-CAM for commfor: {exc}")

            face_entry = {
                "face_index": 0,
                "global_face_id": "Full Frame",
                "prediction_label": pred_label,
                "fake_probability": round(prob_fake, 4),
                "real_probability": round(prob_real, 4),
                "suspicion_level": sev,
                "input_crop_path": crop_path_str,
                "input_size": [cam_w, cam_h],
                "gradcam_overlay_path": gradcam_path_str,
                "padded_ratio": 0.0,
                "unreliable_pose_warning": False,
                "logits": logits_dict,
                "xai_metrics": xai_metrics,
            }

            frame_results.append(
                {
                    "frame_index": frame_idx,
                    "frame_path": frame_path,
                    "evaluated": True,
                    "fake_probability": round(prob_fake, 4),
                    "suspicion_level": sev,
                    "peak_face_id": "Full Frame",
                    "faces": [face_entry],
                }
            )

        if eval_scores:
            mean_s = round(float(np.mean(eval_scores)), 4)
            max_s = round(float(np.max(eval_scores)), 4)
            sorted_desc = sorted(eval_scores, reverse=True)
            k = max(1, math.ceil(0.30 * len(sorted_desc)))
            topk_s = round(float(np.mean(sorted_desc[:k])), 4)
        else:
            mean_s = max_s = topk_s = 0.0

        summary_primary = mean_s if len(frame_results) > 1 else max_s

        return {
            "detector_id": self.detector_id,
            "display_name": self.display_name,
            "paradigm": self.paradigm,
            "input_size": self.input_size,
            "weights_path": self._loaded_weights_path,
            "device_used": str(self.device),
            "frames": frame_results,
            "summary": {
                "frames_evaluated": len(eval_scores),
                "mean_fake_probability": mean_s,
                "max_fake_probability": max_s,
                "topk_fake_probability": topk_s,
                "suspicion_level": compute_suspicion_level(summary_primary),
            },
        }