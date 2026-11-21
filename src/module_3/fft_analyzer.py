from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _imread_bgr_unicode(path: Path) -> Optional[np.ndarray]:
    """Safely read a color BGR image with Unicode path support."""
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


def _crop_uniform_borders(img_bgr: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """
    Remove solid white/black outer margins before full-frame 2D-FFT
    so artificial border edges do not distort the spectrum.
    """
    h, w = img_bgr.shape[:2]
    if h < 48 or w < 48:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
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

    if bottom - top >= 32 and right - left >= 32:
        return img_bgr[top:bottom, left:right]
    return img_bgr


class FFTAnalyzer:
    """
    Module 3.2: Calibrated 2D Fast Fourier Transform & 1/f Spectral Decay Analysis.
    Detects:
      1. Periodic checkerboard spikes from GAN/Deepfake upsampling.
      2. Abnormal 1/f^alpha spectral decay (extreme slope or curve bumps).
      3. Extreme high-frequency luminance/chrominance aliasing.
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

    def _compute_channel_hf_ratio(self, chan: np.ndarray, r_norm: np.ndarray) -> float:
        h, w = chan.shape
        centered = chan - float(np.mean(chan))
        hann_2d = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        f_shift = np.fft.fftshift(np.fft.fft2(centered * hann_2d))
        log_spec = np.log1p(np.abs(f_shift).astype(np.float32))

        low_mid_mask = (r_norm >= 0.05) & (r_norm < self.mid_cutoff)
        high_mask = (r_norm >= self.mid_cutoff) & (r_norm <= 0.95)
        low_mid_e = (
            float(np.mean(log_spec[low_mid_mask])) if np.any(low_mid_mask) else 1e-6
        )
        high_e = float(np.mean(log_spec[high_mask])) if np.any(high_mask) else 0.0
        return float(high_e / max(low_mid_e, 1e-6))

    def analyze(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        save_radial_plot: bool = True,
        trim_borders: bool = False,
    ) -> Dict[str, Any]:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image file not found for 2D-FFT: {src}")

        img_bgr = _imread_bgr_unicode(src)
        if img_bgr is None:
            raise ValueError(f"Unable to read image for 2D-FFT: {src}")

        if trim_borders:
            img_bgr = _crop_uniform_borders(img_bgr)

        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y_chan = ycrcb[:, :, 0]
        cr_chan = ycrcb[:, :, 1]
        cb_chan = ycrcb[:, :, 2]

        h, w = y_chan.shape

        # Subtract mean and apply 2D Hann window to suppress boundary cross artifacts
        y_centered = y_chan - float(np.mean(y_chan))
        hann_2d = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        windowed = y_centered * hann_2d

        f_transform = np.fft.fft2(windowed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift).astype(np.float32)
        log_spec = np.log1p(magnitude)

        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.indices((h, w))
        r_px = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        r_max = max(float(min(cy, cx)), 1.0)
        r_norm = r_px / r_max

        # 1. Luminance & Chrominance HF energy ratios
        low_mid_mask = (r_norm >= 0.05) & (r_norm < self.mid_cutoff)
        high_mask = (r_norm >= self.mid_cutoff) & (r_norm <= 0.95)

        low_mid_energy = (
            float(np.mean(log_spec[low_mid_mask])) if np.any(low_mid_mask) else 1e-6
        )
        high_energy = float(np.mean(log_spec[high_mask])) if np.any(high_mask) else 0.0
        hf_ratio = float(high_energy / max(low_mid_energy, 1e-6))

        if float(np.std(cr_chan)) > 1.0 or float(np.std(cb_chan)) > 1.0:
            chroma_hf_ratio = 0.5 * (
                self._compute_channel_hf_ratio(cr_chan, r_norm)
                + self._compute_channel_hf_ratio(cb_chan, r_norm)
            )
        else:
            chroma_hf_ratio = hf_ratio

        # 2. Periodic spike detection (checkerboard artifacts) outside DC region
        local_bg = cv2.medianBlur(log_spec, 5)
        residual = log_spec - local_bg

        outer_mask = (r_norm >= self.inner_cutoff) & (r_norm <= 0.95)
        outer_res = residual[outer_mask] if np.any(outer_mask) else residual.ravel()
        res_std = float(np.std(outer_res))

        if res_std > 1e-6:
            z_scores = (outer_res - float(np.mean(outer_res))) / res_std
            max_spike_z = float(np.max(z_scores))
            top_spike_mean_z = (
                float(np.mean(np.sort(z_scores)[-20:]))
                if z_scores.size >= 20
                else max_spike_z
            )
            spike_density = float(np.mean(z_scores > 4.0))
        else:
            max_spike_z = 0.0
            top_spike_mean_z = 0.0
            spike_density = 0.0

        # 3. 1D Azimuthally Averaged Radial Power Profile + 1/f^alpha Decay Fit
        max_bin = int(r_max)
        r_int = np.clip(r_px.astype(np.int32), 0, max_bin)
        radial_sum = np.bincount(
            r_int.ravel(), weights=log_spec.ravel(), minlength=max_bin + 1
        )
        radial_cnt = np.bincount(r_int.ravel(), minlength=max_bin + 1)
        radial_profile = radial_sum / np.maximum(radial_cnt, 1)

        start_idx = max(3, int(self.inner_cutoff * max_bin))
        end_idx = max(start_idx + 6, int(0.92 * max_bin))
        tail_profile = radial_profile[start_idx:end_idx]

        radial_roughness = 0.0
        spectral_slope = 1.2
        radial_fit_rmse = 0.0
        fitted = None

        if tail_profile.size >= 6:
            second_diff = np.diff(tail_profile, n=2)
            radial_roughness = float(
                np.std(second_diff) / max(float(np.mean(tail_profile)), 1e-6)
            )
            log_freqs = np.log(
                np.arange(start_idx, start_idx + tail_profile.size, dtype=np.float32)
            )
            slope_m, intercept_b = np.polyfit(
                log_freqs, tail_profile.astype(np.float32), 1
            )
            fitted = slope_m * log_freqs + intercept_b
            spectral_slope = float(-slope_m)
            radial_fit_rmse = float(
                np.sqrt(np.mean((tail_profile - fitted) ** 2))
                / max(float(np.mean(tail_profile)), 1e-6)
            )

        # 4. Calibrated FFT scoring (prevents natural real-video log1p ratios from false-alarming)
        spike_term = float(np.clip((top_spike_mean_z - 3.7) / 4.2, 0.0, 1.0))
        density_term = float(np.clip(spike_density / 0.004, 0.0, 1.0))
        spike_subscore = float(
            np.clip(0.60 * spike_term + 0.40 * density_term, 0.0, 1.0)
        )

        # Small crops (< 100px radius) naturally have slightly higher 1D bin variance
        rough_base = 0.030 if max_bin < 100 else 0.018
        roughness_term = float(
            np.clip((radial_roughness - rough_base) / 0.038, 0.0, 1.0)
        )
        fit_term = float(np.clip((radial_fit_rmse - 0.036) / 0.055, 0.0, 1.0))

        # Natural camera 1/f^alpha slopes lie around [0.75, 2.35]
        if spectral_slope < 0.75:
            slope_term = float(np.clip((0.75 - spectral_slope) / 0.60, 0.0, 1.0))
        elif spectral_slope > 2.35:
            slope_term = float(np.clip((spectral_slope - 2.35) / 1.10, 0.0, 1.0))
        else:
            slope_term = 0.0

        # Real compressed videos often have log1p HF ratios ~0.75-0.89; flag only extreme >0.90
        hf_term = float(np.clip((hf_ratio - 0.90) / 0.12, 0.0, 1.0))
        chroma_hf_term = float(np.clip((chroma_hf_ratio - 0.83) / 0.14, 0.0, 1.0))

        shape_subscore = float(
            np.clip(
                0.32 * fit_term
                + 0.28 * slope_term
                + 0.20 * roughness_term
                + 0.10 * hf_term
                + 0.10 * chroma_hf_term,
                0.0,
                1.0,
            )
        )

        dominant_signal = max(spike_subscore, shape_subscore)
        avg_signal = 0.5 * (spike_subscore + shape_subscore)

        fft_score = round(
            float(np.clip(0.65 * dominant_signal + 0.35 * avg_signal, 0.0, 1.0)),
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
            spec_norm = (
                (log_spec - spec_min) / max(spec_max - spec_min, 1e-6) * 255.0
            ).astype(np.uint8)
            spec_color = cv2.applyColorMap(spec_norm, cv2.COLORMAP_INFERNO)

            spec_file = out_path / f"{tag}_fft_spectrum.png"
            _imwrite_unicode(spec_file, spec_color)
            spectrum_path = str(spec_file.resolve())

            if save_radial_plot and radial_profile.size > 2:
                rad_file = out_path / f"{tag}_fft_radial.png"
                fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
                freqs = np.linspace(0.0, 1.0, radial_profile.size)
                ax.plot(
                    freqs[1:],
                    radial_profile[1:],
                    color="#ff5722",
                    linewidth=1.6,
                    label="Observed Spectrum",
                )
                if fitted is not None and tail_profile.size >= 6:
                    tail_freqs = np.linspace(
                        start_idx / max(max_bin, 1),
                        end_idx / max(max_bin, 1),
                        tail_profile.size,
                    )
                    ax.plot(
                        tail_freqs,
                        fitted,
                        color="#2196f3",
                        linestyle=":",
                        linewidth=1.3,
                        label="Natural 1/f Trend",
                    )
                ax.axvline(
                    self.mid_cutoff,
                    color="#607d8b",
                    linestyle="--",
                    linewidth=1.0,
                    label="High-Freq Cutoff",
                )
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
                "chroma_hf_ratio": round(chroma_hf_ratio, 4),
                "max_spike_z": round(max_spike_z, 4),
                "top20_spike_mean_z": round(top_spike_mean_z, 4),
                "spike_density": round(spike_density, 6),
                "radial_roughness": round(radial_roughness, 5),
                "radial_fit_rmse": round(radial_fit_rmse, 5),
                "spectral_slope": round(spectral_slope, 4),
            },
            "fft_spectrum_path": spectrum_path,
            "fft_radial_plot_path": radial_plot_path,
        }
