from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _imread_gray_unicode(path: Path) -> Optional[np.ndarray]:
    """Safely read a grayscale image with Unicode path support."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size > 0:
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    try:
        with Image.open(path) as pil_img:
            return np.asarray(pil_img.convert("L"), dtype=np.uint8)
    except Exception:
        return None


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


class FFTAnalyzer:
    """
    Module 3.2: 2D Fast Fourier Transform (2D-FFT)
    Detects periodic checkerboard artifacts caused by GAN/Deepfake upsampling
    and abnormal high-frequency power spectrum roll-off.
    """

    def __init__(self, inner_cutoff: float = 0.12, mid_cutoff: float = 0.50):
        self.inner_cutoff = float(inner_cutoff)
        self.mid_cutoff = float(mid_cutoff)

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
        prefix: Optional[str] = None,
        save_radial_plot: bool = True,
    ) -> Dict[str, Any]:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image file not found for 2D-FFT: {src}")

        gray = _imread_gray_unicode(src)
        if gray is None:
            raise ValueError(f"Unable to read image for 2D-FFT: {src}")

        gray_f = gray.astype(np.float32)
        h, w = gray_f.shape

        # Subtract mean and apply a 2D Hann window to suppress boundary cross artifacts
        gray_centered = gray_f - float(np.mean(gray_f))
        hann_2d = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        windowed = gray_centered * hann_2d

        # Compute 2D-FFT and shift DC component to center
        f_transform = np.fft.fft2(windowed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift).astype(np.float32)
        log_spec = np.log1p(magnitude)

        # Normalized radial coordinate r in [0, 1] from DC center
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.indices((h, w))
        r_px = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        r_max = max(float(min(cy, cx)), 1.0)
        r_norm = r_px / r_max

        # 1. High-frequency to low/mid-frequency energy ratio
        low_mid_mask = (r_norm >= 0.05) & (r_norm < self.mid_cutoff)
        high_mask = (r_norm >= self.mid_cutoff) & (r_norm <= 0.95)

        low_mid_energy = float(np.mean(log_spec[low_mid_mask])) if np.any(low_mid_mask) else 1e-6
        high_energy = float(np.mean(log_spec[high_mask])) if np.any(high_mask) else 0.0
        hf_ratio = float(high_energy / max(low_mid_energy, 1e-6))

        # 2. Periodic spike detection (checkerboard artifacts) outside DC region
        # Use 5x5 median blur to estimate smooth spectral background and isolate sharp spikes
        local_bg = cv2.medianBlur(log_spec, 5)
        residual = log_spec - local_bg

        outer_mask = (r_norm >= self.inner_cutoff) & (r_norm <= 0.95)
        outer_res = residual[outer_mask] if np.any(outer_mask) else residual.ravel()
        res_std = float(np.std(outer_res))

        if res_std > 1e-6:
            z_scores = (outer_res - float(np.mean(outer_res))) / res_std
            max_spike_z = float(np.max(z_scores))
            top_spike_mean_z = float(np.mean(np.sort(z_scores)[-20:])) if z_scores.size >= 20 else max_spike_z
            spike_density = float(np.mean(z_scores > 4.0))
        else:
            max_spike_z = 0.0
            top_spike_mean_z = 0.0
            spike_density = 0.0

        # 3. 1D Azimuthally Averaged Radial Power Profile
        max_bin = int(r_max)
        r_int = np.clip(r_px.astype(np.int32), 0, max_bin)
        radial_sum = np.bincount(r_int.ravel(), weights=log_spec.ravel(), minlength=max_bin + 1)
        radial_cnt = np.bincount(r_int.ravel(), minlength=max_bin + 1)
        radial_profile = radial_sum / np.maximum(radial_cnt, 1)

        # Curve roughness outside DC region -> sensitive to periodic noise
        start_idx = max(2, int(self.inner_cutoff * max_bin))
        end_idx = max(start_idx + 4, int(0.95 * max_bin))
        tail_profile = radial_profile[start_idx:end_idx]
        if tail_profile.size >= 4:
            second_diff = np.diff(tail_profile, n=2)
            radial_roughness = float(np.std(second_diff) / max(float(np.mean(tail_profile)), 1e-6))
        else:
            radial_roughness = 0.0

        # 4. Normalize 2D-FFT suspicion score to [0.0, 1.0] for Late Fusion (Module 5)
        spike_term = float(np.clip((top_spike_mean_z - 3.8) / 4.5, 0.0, 1.0))
        density_term = float(np.clip(spike_density / 0.005, 0.0, 1.0))
        hf_term = float(np.clip((hf_ratio - 0.58) / 0.30, 0.0, 1.0))
        roughness_term = float(np.clip((radial_roughness - 0.015) / 0.045, 0.0, 1.0))

        fft_score = round(
            float(
                np.clip(
                    0.40 * spike_term
                    + 0.25 * density_term
                    + 0.20 * hf_term
                    + 0.15 * roughness_term,
                    0.0,
                    1.0,
                )
            ),
            4,
        )

        # 5. Export 2D-FFT spectrum image and 1D radial profile plot
        spectrum_path: Optional[str] = None
        radial_plot_path: Optional[str] = None

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            tag = prefix or src.stem

            spec_min, spec_max = float(np.min(log_spec)), float(np.max(log_spec))
            spec_norm = ((log_spec - spec_min) / max(spec_max - spec_min, 1e-6) * 255.0).astype(np.uint8)
            spec_color = cv2.applyColorMap(spec_norm, cv2.COLORMAP_INFERNO)

            spec_file = out_path / f"{tag}_fft_spectrum.png"
            _imwrite_unicode(spec_file, spec_color)
            spectrum_path = str(spec_file.resolve())

            if save_radial_plot and radial_profile.size > 2:
                rad_file = out_path / f"{tag}_fft_radial.png"
                fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
                freqs = np.linspace(0.0, 1.0, radial_profile.size)
                ax.plot(freqs[1:], radial_profile[1:], color="#ff5722", linewidth=1.6)
                ax.axvline(self.mid_cutoff, color="#607d8b", linestyle="--", linewidth=1.0, label="High-Freq Cutoff")
                ax.set_title("1D Azimuthal Power Spectrum (2D-FFT)")
                ax.set_xlabel("Normalized Spatial Frequency (r / r_max)")
                ax.set_ylabel("Log Magnitude")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                fig.savefig(str(rad_file))
                plt.close(fig)
                radial_plot_path = str(rad_file.resolve())

        return {
            "source_path": str(src.resolve()),
            "image_resolution": [int(w), int(h)],
            "fft_score": fft_score,
            "suspicion_level": self._compute_suspicion_level(fft_score),
            "metrics": {
                "hf_energy_ratio": round(hf_ratio, 4),
                "max_spike_z": round(max_spike_z, 4),
                "top20_spike_mean_z": round(top_spike_mean_z, 4),
                "spike_density": round(spike_density, 6),
                "radial_roughness": round(radial_roughness, 5),
            },
            "fft_spectrum_path": spectrum_path,
            "fft_radial_plot_path": radial_plot_path,
        }
