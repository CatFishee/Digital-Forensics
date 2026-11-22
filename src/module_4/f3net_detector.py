from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModule4Detector, compute_suspicion_level, resolve_torch_device


def _imwrite_unicode(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def _dct_matrix(size: int) -> np.ndarray:
    """Orthonormal 1D DCT-II basis matrix of shape (size, size)."""
    i = np.arange(size, dtype=np.float32)[:, None]
    j = np.arange(size, dtype=np.float32)[None, :]
    scale = np.where(i == 0, math.sqrt(1.0 / size), math.sqrt(2.0 / size))
    return (scale * np.cos((j + 0.5) * math.pi * i / size)).astype(np.float32)


def _generate_band_mask(start: float, end: float, size: int) -> np.ndarray:
    """Triangular frequency-band binary mask over (i + j) DCT coordinates."""
    idx = np.arange(size, dtype=np.float32)
    diag = idx[:, None] + idx[None, :]
    mask = np.where((diag < start) | (diag > end), 0.0, 1.0)
    return mask.astype(np.float32)


def _norm_sigma(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sigmoid(x) - 1.0


class FrequencyFilter(nn.Module):
    """Learnable frequency-band filter matching DeepfakeBench F3Net Filter."""

    def __init__(
        self,
        size: int,
        band_start: float,
        band_end: float,
        use_learnable: bool = True,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_learnable = use_learnable
        base_np = _generate_band_mask(band_start, band_end, size)
        self.base = nn.Parameter(torch.from_numpy(base_np), requires_grad=False)

        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0.0, 0.1)

        self.norm = norm
        if self.norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.from_numpy(base_np)), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_learnable:
            filt = self.base + _norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            return x * filt / self.ft_num
        return x * filt


class FADHead(nn.Module):
    """
    DeepfakeBench FAD_Head:
      RGB [B, 3, H, W] -> 2D-DCT -> 4 learnable bands (low, mid, high, all) -> IDCT -> [B, 12, H, W]
    """

    def __init__(self, size: int = 256) -> None:
        super().__init__()
        dct_np = _dct_matrix(size)
        self._DCT_all = nn.Parameter(torch.from_numpy(dct_np), requires_grad=False)
        self._DCT_all_T = nn.Parameter(
            torch.from_numpy(dct_np.T.copy()), requires_grad=False
        )

        low_filter = FrequencyFilter(size, 0, size // 2.82)
        middle_filter = FrequencyFilter(size, size // 2.82, size // 2)
        high_filter = FrequencyFilter(size, size // 2, size * 2)
        all_filter = FrequencyFilter(size, 0, size * 2)

        self.filters = nn.ModuleList(
            [low_filter, middle_filter, high_filter, all_filter]
        )

    def forward_bands(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y_list: List[torch.Tensor] = []
        for filt in self.filters:
            x_pass = filt(x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        return torch.cat(y_list, dim=1), y_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_bands(x)
        return out


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """
    Matches DeepfakeBench / Cadene Xception Block state_dict indices identically,
    using non-shared nn.ReLU(inplace=False) so PyTorch backward hooks for Grad-CAM never crash.
    """

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(
                in_filters, out_filters, 1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep: List[nn.Module] = []
        filters = in_filters

        if grow_first:
            rep.append(nn.ReLU(inplace=False))
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU(inplace=False))
            rep.append(
                SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False)
            )
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=False))
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        return x + skip


class Xception12Backbone(nn.Module):
    """
    DeepfakeBench 12-channel Xception backbone for F3Net.
    Matches DeepfakeBench's `adjust_channel` + `last_linear` Sequential structure
    and uses non-inplace ReLUs so Grad-CAM can hook `act4` cleanly.
    """

    def __init__(
        self,
        num_classes: int = 2,
        inc: int = 12,
        dropout: bool = False,
        mode: str = "original",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode = mode

        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=False)

        self.block1 = XceptionBlock(
            64, 128, 2, 2, start_with_relu=False, grow_first=True
        )
        self.block2 = XceptionBlock(
            128, 256, 2, 2, start_with_relu=True, grow_first=True
        )
        self.block3 = XceptionBlock(
            256, 728, 2, 2, start_with_relu=True, grow_first=True
        )

        self.block4 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = XceptionBlock(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False
        )

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=False)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.act4 = nn.ReLU(inplace=False)

        # Always instantiated in DeepfakeBench Xception; applied in features() when mode == "adjust_channel"
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

        in_feat = 512 if self.mode == "adjust_channel" else 2048
        self.last_linear = nn.Sequential(
            nn.Dropout(p=0.5) if self.dropout else nn.Identity(),
            nn.Linear(in_feat, num_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, feat: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(feat, (1, 1))
        x = x.view(x.size(0), -1)
        return self.last_linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class F3NetModel(nn.Module):
    """
    Standalone DeepfakeBench F3Net (FAD branch + 12-channel Xception).
    Accepts tensor [B, 3, 256, 256] normalized to [-1, 1] and outputs [B, 2] logits.
    """

    def __init__(
        self, img_size: int = 256, num_classes: int = 2, mode: str = "original"
    ) -> None:
        super().__init__()
        self.FAD_head = FADHead(size=img_size)
        self.backbone = Xception12Backbone(
            num_classes=num_classes, inc=12, dropout=False, mode=mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea_fad = self.FAD_head(x)
        feat = self.backbone.features(fea_fad)
        return self.backbone.classifier(feat)


def _extract_state_dict(raw_ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw_ckpt, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in raw_ckpt and isinstance(raw_ckpt[key], dict):
                raw_ckpt = raw_ckpt[key]
                break
    if not isinstance(raw_ckpt, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict dict")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in raw_ckpt.items():
        if not isinstance(v, torch.Tensor):
            continue
        new_k = k[7:] if k.startswith("module.") else k
        if new_k.startswith("loss_func."):
            continue
        if new_k in ("backbone.last_linear.weight", "backbone.last_linear.bias"):
            suffix = new_k.rsplit(".", 1)[-1]
            new_k = f"backbone.last_linear.1.{suffix}"
        elif new_k in ("backbone.fc.weight", "backbone.fc.bias"):
            suffix = new_k.rsplit(".", 1)[-1]
            new_k = f"backbone.last_linear.1.{suffix}"
        if "pointwise" in new_k and v.ndim == 2:
            v = v.unsqueeze(-1).unsqueeze(-1)
        cleaned[new_k] = v
    return cleaned


class F3NetDetector(BaseModule4Detector):
    """
    Module 4 Detector Plugin: F3Net (DeepfakeBench FAD-DCT + Xception).
    """

    detector_id = "f3net"
    display_name = "F3Net (DeepfakeBench FAD-DCT)"
    paradigm = "frequency_dct"
    input_size = 256
    input_source = "aligned_face"
    requires_face = True

    def __init__(self, weights_path: Optional[str] = None) -> None:
        super().__init__(weights_path=weights_path)
        self.model: Optional[F3NetModel] = None
        self.loaded_weights_path: Optional[str] = None

    def load(self, device: str = "auto", auto_download: bool = True) -> None:
        if self._is_loaded and self.model is not None:
            return

        ckpt_path = self.resolve_weights_path(auto_download=auto_download)
        self.device = resolve_torch_device(device)

        try:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state_dict = _extract_state_dict(raw_ckpt)
        head_w = state_dict.get("backbone.last_linear.1.weight")
        inferred_mode = (
            "adjust_channel"
            if head_w is not None and head_w.ndim == 2 and head_w.shape[1] == 512
            else "original"
        )

        model = F3NetModel(img_size=self.input_size, num_classes=2, mode=inferred_mode)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"F3Net checkpoint mismatch at '{ckpt_path}'. "
                f"Missing keys: {missing[:8]} | Unexpected keys: {unexpected[:8]}"
            )

        model.to(self.device)
        model.eval()

        self.model = model
        self.loaded_weights_path = ckpt_path
        self._is_loaded = True

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        super().unload()

    def _preprocess_face(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load aligned PNG as RGB, resize to 256x256 with Bicubic if needed,
        and normalize to [-1, 1] (DeepfakeBench mean=0.5, std=0.5).
        Returns (tensor [3, 256, 256], rgb_float_01 [256, 256, 3]).
        """
        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            if rgb_img.size != (self.input_size, self.input_size):
                rgb_img = rgb_img.resize(
                    (self.input_size, self.input_size), Image.Resampling.BICUBIC
                )
            rgb_u8 = np.asarray(rgb_img, dtype=np.uint8)

        rgb_01 = rgb_u8.astype(np.float32) / 255.0
        norm = (rgb_01 - 0.5) / 0.5
        tensor = torch.from_numpy(norm).permute(2, 0, 1).contiguous()
        return tensor, rgb_01

    def _compute_band_energy_shares(self, x_batch: torch.Tensor) -> List[Dict[str, float]]:
        """
        Compute relative spatial energy across FAD's Low, Mid, and High frequency bands
        to enrich frequency explainability alongside Grad-CAM.
        """
        assert self.model is not None
        with torch.no_grad():
            _, band_tensors = self.model.FAD_head.forward_bands(x_batch)
            e_low = torch.mean(torch.abs(band_tensors[0]), dim=(1, 2, 3))
            e_mid = torch.mean(torch.abs(band_tensors[1]), dim=(1, 2, 3))
            e_high = torch.mean(torch.abs(band_tensors[2]), dim=(1, 2, 3))
            total = (e_low + e_mid + e_high).clamp(min=1e-6)

            shares: List[Dict[str, float]] = []
            for idx in range(x_batch.shape[0]):
                shares.append(
                    {
                        "low_band_share": round(float((e_low[idx] / total[idx]).item()), 4),
                        "mid_band_share": round(float((e_mid[idx] / total[idx]).item()), 4),
                        "high_band_share": round(float((e_high[idx] / total[idx]).item()), 4),
                    }
                )
            return shares

    def analyze_case(
        self,
        m2_manifest: Dict[str, Any],
        output_dir: str,
        batch_size: int = 4,
        enable_gradcam: bool = True,
    ) -> Dict[str, Any]:
        if not self._is_loaded or self.model is None:
            raise RuntimeError("F3NetDetector.load() must be called before analyze_case().")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        frames_detail = m2_manifest.get("frames_detail", []) or []

        face_jobs: List[Dict[str, Any]] = []
        for frame_pos, frame_rec in enumerate(frames_detail):
            frame_idx = int(frame_rec.get("frame_display_index", frame_pos + 1))
            frame_stem = Path(frame_rec.get("frame_path", f"frame_{frame_idx:05d}")).stem
            for f_info in frame_rec.get("faces", []) or []:
                aligned_path = f_info.get("aligned_face_path")
                if not aligned_path or not Path(aligned_path).exists():
                    continue
                face_jobs.append(
                    {
                        "frame_pos": frame_pos,
                        "frame_index": frame_idx,
                        "frame_stem": frame_stem,
                        "face_index": int(f_info.get("face_index", 0)),
                        "global_face_id": f_info.get("global_face_id")
                        or f"Face #{f_info.get('face_index', 0)}",
                        "aligned_face_path": aligned_path,
                    }
                )

        face_results_by_frame: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(len(frames_detail))
        }

        if face_jobs:
            target_layer = (
                self.model.backbone.adjust_channel
                if self.model.backbone.mode == "adjust_channel"
                else self.model.backbone.act4
            )
            cam_ctx = (
                GradCAM(
                    model=self.model,
                    target_layers=[target_layer],
                )
                if enable_gradcam
                else None
            )

            try:
                step = max(1, int(batch_size))
                for start in range(0, len(face_jobs), step):
                    chunk = face_jobs[start : start + step]
                    tensors: List[torch.Tensor] = []
                    rgb_01_list: List[np.ndarray] = []

                    for job in chunk:
                        t, rgb_01 = self._preprocess_face(job["aligned_face_path"])
                        tensors.append(t)
                        rgb_01_list.append(rgb_01)

                    x_batch = torch.stack(tensors, dim=0).to(self.device)

                    with torch.no_grad():
                        logits = self.model(x_batch)
                        probs = torch.softmax(logits, dim=1)
                    band_shares = self._compute_band_energy_shares(x_batch)

                    cam_maps: Optional[np.ndarray] = None
                    if cam_ctx is not None:
                        targets = [ClassifierOutputTarget(1) for _ in range(len(chunk))]
                        with torch.enable_grad():
                            cam_maps = cam_ctx(input_tensor=x_batch, targets=targets)

                    for i, job in enumerate(chunk):
                        fake_prob = round(float(probs[i, 1].item()), 4)
                        real_prob = round(float(probs[i, 0].item()), 4)
                        fake_logit = round(float(logits[i, 1].item()), 4)
                        real_logit = round(float(logits[i, 0].item()), 4)

                        gradcam_path_str: Optional[str] = None
                        if cam_maps is not None:
                            cam_2d = np.nan_to_num(
                                cam_maps[i], nan=0.0, posinf=1.0, neginf=0.0
                            )
                            cam_2d = np.clip(cam_2d, 0.0, 1.0)
                            overlay_rgb = show_cam_on_image(
                                rgb_01_list[i], cam_2d, use_rgb=True, image_weight=0.55
                            )
                            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

                            clean_gid = (
                                str(job["global_face_id"])
                                .replace(" ", "_")
                                .replace("#", "")
                            )
                            cam_file = (
                                out_path
                                / f"{job['frame_stem']}_{clean_gid}_f3net_gradcam.png"
                            )
                            if _imwrite_unicode(cam_file, overlay_bgr):
                                gradcam_path_str = str(cam_file.resolve())

                        face_results_by_frame[job["frame_pos"]].append(
                            {
                                "face_index": job["face_index"],
                                "global_face_id": job["global_face_id"],
                                "aligned_face_path": job["aligned_face_path"],
                                "fake_probability": fake_prob,
                                "real_probability": real_prob,
                                "prediction_label": "FAKE" if fake_prob >= 0.5 else "REAL",
                                "suspicion_level": compute_suspicion_level(fake_prob),
                                "logits": {"real": real_logit, "fake": fake_logit},
                                "fad_band_energy": band_shares[i],
                                "gradcam_overlay_path": gradcam_path_str,
                            }
                        )
            finally:
                if cam_ctx is not None:
                    cam_ctx.activations_and_grads.release()

        frame_records: List[Dict[str, Any]] = []
        evaluated_scores: List[float] = []

        for frame_pos, frame_rec in enumerate(frames_detail):
            frame_idx = int(frame_rec.get("frame_display_index", frame_pos + 1))
            f_list = face_results_by_frame.get(frame_pos, [])
            if f_list:
                peak_face = max(f_list, key=lambda x: float(x["fake_probability"]))
                frame_score = float(peak_face["fake_probability"])
                evaluated_scores.append(frame_score)
                frame_records.append(
                    {
                        "frame_index": frame_idx,
                        "frame_path": frame_rec.get("frame_path"),
                        "evaluated": True,
                        "fake_probability": round(frame_score, 4),
                        "suspicion_level": compute_suspicion_level(frame_score),
                        "peak_face_id": peak_face["global_face_id"],
                        "faces": f_list,
                    }
                )
            else:
                frame_records.append(
                    {
                        "frame_index": frame_idx,
                        "frame_path": frame_rec.get("frame_path"),
                        "evaluated": False,
                        "fake_probability": None,
                        "suspicion_level": "low",
                        "peak_face_id": None,
                        "faces": [],
                    }
                )

        if evaluated_scores:
            mean_score = round(float(np.mean(evaluated_scores)), 4)
            max_score = round(float(np.max(evaluated_scores)), 4)
            sorted_desc = sorted(evaluated_scores, reverse=True)
            k = max(1, math.ceil(0.30 * len(sorted_desc)))
            topk_score = round(float(np.mean(sorted_desc[:k])), 4)
        else:
            mean_score = 0.0
            max_score = 0.0
            topk_score = 0.0

        return {
            "detector_id": self.detector_id,
            "display_name": self.display_name,
            "paradigm": self.paradigm,
            "input_size": self.input_size,
            "weights_path": self.loaded_weights_path,
            "device_used": str(self.device),
            "frames": frame_records,
            "summary": {
                "frames_evaluated": len(evaluated_scores),
                "faces_evaluated": sum(len(fr["faces"]) for fr in frame_records),
                "mean_fake_probability": mean_score,
                "max_fake_probability": max_score,
                "topk_fake_probability": topk_score,
                "suspicion_level": compute_suspicion_level(max_score),
            },
        }
