from pathlib import Path
from typing import Any, Dict, List


class Module3ForensicTranslator:
    """
    Deterministic Rule-Based Forensic Translator for Module 3
    (Noise Residual & Boundary Analysis, Calibrated 2D-FFT, and JPEG ELA).
    Metric keys match NoiseAnalyzer, FFTAnalyzer, and ELAAnalyzer 1:1.
    """

    @staticmethod
    def _severity_label(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    @staticmethod
    def _friendly_level(severity: str) -> str:
        sev = str(severity).lower()
        if sev == "high":
            return "High Suspicion"
        if sev == "medium":
            return "Moderate Suspicion"
        return "Low Suspicion"

    @staticmethod
    def _pct(score: float) -> str:
        return f"{float(score) * 100:.0f}%"

    def interpret_noise(self, noise_res: Dict[str, Any]) -> Dict[str, Any]:
        noise_score = float(noise_res.get("noise_score", 0.0))
        suspicion = str(
            noise_res.get("suspicion_level", self._severity_label(noise_score))
        ).lower()
        level_label = self._friendly_level(suspicion)
        pct = self._pct(noise_score)
        metrics = noise_res.get("metrics", {}) or {}

        p95_p50 = float(metrics.get("noise_p95_p50_ratio", 1.0))
        block_cv = float(metrics.get("noise_block_cv", 0.0))
        face_noise = metrics.get("face_noise_median")
        bg_noise = metrics.get("bg_noise_median")
        face_contrast = metrics.get("face_bg_noise_contrast")
        face_smoother = metrics.get("face_smoother_than_bg")
        chroma_contrast = metrics.get("face_bg_chroma_contrast")
        seam_ratio = metrics.get("boundary_seam_ratio")

        findings: List[Dict[str, str]] = []

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Strong noise or blending mismatches were found. "
                "The skin grain inside the face or along its edges differs sharply from the rest of the frame, "
                "which can happen in face-swaps or localized AI edits."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate noise or blending differences appear in this frame. "
                "The face texture or perimeter behaves somewhat differently from the surrounding scene."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Pixel grain looks consistent across the frame. "
                "No strong face-swap blending seams or noise-floor mismatches stood out."
            )

        # 1. Face vs Background Skin Noise Floor
        if face_contrast is not None and face_noise is not None and bg_noise is not None:
            c_val = float(face_contrast)
            direction_text = (
                "smoother (less natural grain) than the background"
                if face_smoother
                else "noisier or sharper than the background"
            )
            if c_val >= 0.90:
                findings.append(
                    {
                        "title": "Strong Noise Mismatch Between Face Skin and Scene",
                        "severity": "alert",
                        "detail": (
                            f"The face skin looks noticeably {direction_text}. "
                            "Real camera scenes usually share a consistent sensor grain; pasted or AI-generated faces often do not."
                        ),
                    }
                )
            elif c_val >= 0.45:
                findings.append(
                    {
                        "title": "Moderate Noise Difference Between Face Skin and Scene",
                        "severity": "warning",
                        "detail": (
                            f"The face skin appears somewhat {direction_text}. "
                            "This can happen with face-swaps, beauty smoothing, or strong background blur."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Consistent Face and Background Noise",
                        "severity": "ok",
                        "detail": "The fine pixel grain on the face skin matches the surrounding scene well.",
                    }
                )
        else:
            findings.append(
                {
                    "title": "No Face Mask Comparison",
                    "severity": "info",
                    "detail": "No valid face box was available, so only global frame noise uniformity was checked.",
                }
            )

        # 2. Color-channel noise check
        if chroma_contrast is not None and float(chroma_contrast) >= 0.60:
            findings.append(
                {
                    "title": "Color-Grain Mismatch Inside Face",
                    "severity": "warning",
                    "detail": (
                        "Fine color noise inside the face differs noticeably from the rest of the frame, "
                        "which can occur when a face is inserted from a different source or AI model."
                    ),
                }
            )

        # 3. Face perimeter blending seam check
        if seam_ratio is not None:
            sr_val = float(seam_ratio)
            if sr_val >= 2.25:
                findings.append(
                    {
                        "title": "Possible Face-Border Blending Seam",
                        "severity": "alert",
                        "detail": (
                            "Unusual texture transitions appear right along the outer border of the face, "
                            "which can indicate a pasted or feathered face-swap mask."
                        ),
                    }
                )
            elif sr_val >= 1.70:
                findings.append(
                    {
                        "title": "Mild Edge Transition Around Face",
                        "severity": "warning",
                        "detail": (
                            "A slight texture transition appears around the face perimeter; "
                            "check the heatmap along the jawline and cheeks."
                        ),
                    }
                )

        # 4. Global noise uniformity
        if p95_p50 >= 6.0 or block_cv >= 1.35:
            findings.append(
                {
                    "title": "Uneven Noise Patches Across Frame",
                    "severity": "warning",
                    "detail": (
                        "Some patches of the frame have noticeably heavier or smoother grain than others, "
                        "which can point to pasted regions, graphic overlays, or selective retouching."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "Steady Background Grain",
                    "severity": "ok",
                    "detail": "Noise levels across patches stay within a normal natural range.",
                }
            )

        user_guidance = (
            "How to read the images above: The left image strips away colors and lighting to show only tiny pixel grain; "
            "the heatmap on the right shows smooth/low-grain areas in dark blue and high-grain areas in yellow-red. "
            "Look for an unnatural contrast between smooth face skin and the neck/background."
        )

        return {
            "headline": headline,
            "severity": suspicion,
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_ela(self, ela_res: Dict[str, Any]) -> Dict[str, Any]:
        ela_score = float(ela_res.get("ela_score", 0.0))
        suspicion = str(
            ela_res.get("suspicion_level", self._severity_label(ela_score))
        ).lower()
        level_label = self._friendly_level(suspicion)
        pct = self._pct(ela_score)

        is_jpeg = bool(ela_res.get("is_jpeg", False))
        orig_fmt = str(ela_res.get("original_format", "UNKNOWN")).upper()
        metrics = ela_res.get("metrics", {}) or {}

        p95_p50 = float(metrics.get("block_p95_p50_ratio", 1.0))
        block_cv = float(metrics.get("block_cv", 0.0))
        hotspot_ratio = float(metrics.get("hotspot_ratio", 0.0))
        face_mean = metrics.get("face_region_mean")
        bg_mean = metrics.get("bg_region_mean")
        face_contrast = metrics.get("face_bg_contrast")

        findings: List[Dict[str, str]] = []

        if not is_jpeg:
            headline = (
                f"Not Applicable for Scoring ({orig_fmt} Source) — This frame is not a native JPEG photo, "
                "so ELA is shown for reference only and excluded from the Module 3 score."
            )
        elif suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Noticeable compression differences were found across this JPEG. "
                "Some areas react very differently when re-saved, which can happen when faces or objects are pasted or edited."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Some compression differences appear across this JPEG. "
                "Certain areas react differently when re-saved, so a closer visual check is recommended."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Compression looks consistent across this JPEG. "
                "No strong signs of pasted or heavily re-edited patches were found."
            )

        if is_jpeg:
            findings.append(
                {
                    "title": "Original Photo Format (JPEG)",
                    "severity": "ok",
                    "detail": "This image is a native JPEG photo, so compression history checks are reliable.",
                }
            )
        else:
            findings.append(
                {
                    "title": f"Non-JPEG Source ({orig_fmt}) — Excluded from Score",
                    "severity": "info",
                    "detail": (
                        f"Extracted video frames and {orig_fmt} files do not preserve original JPEG block history. "
                        "Use the Noise & Boundary and 2D-FFT tabs for reliable signal results on this file."
                    ),
                }
            )

        if face_contrast is not None and face_mean is not None and bg_mean is not None:
            face_val = float(face_mean)
            bg_val = float(bg_mean)
            contrast_val = float(face_contrast)
            direction = (
                "reacts more strongly when re-saved than the background"
                if face_val > bg_val
                else "looks smoother or more compressed than the background"
            )

            if contrast_val >= 0.65:
                findings.append(
                    {
                        "title": "Clear Difference Between Face and Background",
                        "severity": "alert",
                        "detail": f"The face area {direction}.",
                    }
                )
            elif contrast_val >= 0.35:
                findings.append(
                    {
                        "title": "Noticeable Difference Between Face and Background",
                        "severity": "warning",
                        "detail": f"The face area {direction}.",
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Face and Background Match Well",
                        "severity": "ok",
                        "detail": "The face area and background react similarly when re-saved.",
                    }
                )

        if p95_p50 >= 3.2 or block_cv >= 0.90:
            findings.append(
                {
                    "title": "Uneven Compression Patches",
                    "severity": "warning",
                    "detail": (
                        "Some small patches react much more strongly than the rest of the image. "
                        "This often shows up around edited spots, added text, or sharp overlays."
                    ),
                }
            )

        if hotspot_ratio >= 0.05:
            findings.append(
                {
                    "title": "Bright Hotspots Present",
                    "severity": "warning",
                    "detail": (
                        "Sharp edges like text, watermarks, or hair naturally glow bright—focus on whether "
                        "smooth skin looks noticeably different from plain background areas."
                    ),
                }
            )

        user_guidance = (
            "How to read the images above: Bright areas show pixels that changed most when re-saved. "
            "Sharp edges like text and hair naturally glow bright; look instead for smooth areas like cheeks "
            "that stand out sharply against plain background areas."
        )

        return {
            "headline": headline,
            "severity": suspicion if is_jpeg else "low",
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_fft(self, frame_m3: Dict[str, Any]) -> Dict[str, Any]:
        full_fft = frame_m3.get("fft_full_frame", {}) or {}
        face_ffts = frame_m3.get("fft_unwarped_faces", []) or []
        summary = frame_m3.get("summary", {}) or {}

        frame_score = float(summary.get("frame_fft_score", full_fft.get("fft_score", 0.0)))
        max_face_score = summary.get("max_face_fft_score")
        effective_score = float(summary.get("effective_fft_score", frame_score))
        suspicion = str(
            summary.get("fft_suspicion", self._severity_label(effective_score))
        ).lower()
        level_label = self._friendly_level(suspicion)
        pct = self._pct(effective_score)

        findings: List[Dict[str, str]] = []

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Strong frequency anomalies were detected. "
                "Unusual repeating grid dots or abnormal spectral decay appear, "
                "which commonly come from AI generation, upscaling, or face-swapping."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate frequency irregularities were found. "
                "Some fine-detail patterns deviate from a typical natural camera curve."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Frequency patterns look natural. "
                "No strong repeating AI grid traces or abnormal spectral deviations were found."
            )

        primary_label = "Full Frame"
        primary_fft = full_fft
        if face_ffts:
            top_face = max(face_ffts, key=lambda x: float(x.get("fft_score", 0.0)))
            if float(top_face.get("fft_score", 0.0)) >= frame_score:
                primary_label = str(top_face.get("global_face_id", "Detected Face"))
                primary_fft = top_face

        pm = primary_fft.get("metrics", {}) or {}
        top20_z = float(pm.get("top20_spike_mean_z", 0.0))
        max_z = float(pm.get("max_spike_z", 0.0))
        spike_density = float(pm.get("spike_density", 0.0))
        hf_ratio = float(pm.get("hf_energy_ratio", 0.0))
        chroma_hf = float(pm.get("chroma_hf_ratio", 0.0))
        roughness = float(pm.get("radial_roughness", 0.0))
        fit_rmse = float(pm.get("radial_fit_rmse", 0.0))
        slope = float(pm.get("spectral_slope", 1.2))

        # 1. Periodic checkerboard grid dots
        if top20_z >= 5.2 or max_z >= 7.0:
            findings.append(
                {
                    "title": f"Repeating Grid Pattern Found ({primary_label})",
                    "severity": "alert",
                    "detail": (
                        "Unusual repeating bright dots appear in the frequency scan. "
                        "AI generators and face-swap tools often leave faint grid patterns like these."
                    ),
                }
            )
        elif top20_z >= 3.7 or spike_density >= 0.002:
            findings.append(
                {
                    "title": f"Mild Repeating Patterns ({primary_label})",
                    "severity": "warning",
                    "detail": (
                        "A few small repeating spots appear in the frequency scan, "
                        "which can come from AI upscaling, resizing, or regular textures."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "No Repeating AI Grid Dots Found",
                    "severity": "ok",
                    "detail": (
                        "No sharp repeating grid dots stood out. Note that resizing or JPEG/video compression "
                        "can smooth out faint grid dots."
                    ),
                }
            )

        # 2. 1/f spectral decay curve & slope check
        if (
            fit_rmse >= 0.050
            or slope < 0.70
            or slope > 2.45
            or roughness >= 0.036
            or hf_ratio >= 0.92
            or chroma_hf >= 0.86
        ):
            findings.append(
                {
                    "title": f"Abnormal Fine-Detail Curve ({primary_label})",
                    "severity": "warning",
                    "detail": (
                        "The orange spectrum curve bends away from a normal natural camera slope. "
                        "This can point to synthetic textures, upscaling, heavy over-smoothing, or artificial sharpening."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "Natural Detail Roll-Off",
                    "severity": "ok",
                    "detail": "The frequency curve follows a smooth, natural camera decay from left to right.",
                }
            )

        # 3. Face vs Full frame
        if max_face_score is not None:
            face_val = float(max_face_score)
            if face_val >= frame_score + 0.15:
                findings.append(
                    {
                        "title": "Anomaly Concentrated Inside Face",
                        "severity": "warning",
                        "detail": (
                            f"The zoomed-in face scan ({self._pct(face_val)}) scores higher than the full frame ({self._pct(frame_score)}), "
                            "pointing to unusual patterns inside the face region itself."
                        ),
                    }
                )
            elif frame_score >= face_val + 0.15:
                findings.append(
                    {
                        "title": "Full Frame Higher Than Face",
                        "severity": "info",
                        "detail": (
                            f"The full frame ({self._pct(frame_score)}) scores higher than the isolated face ({self._pct(face_val)}), "
                            "which can happen when background textures, banners, or graphic edges add high-frequency detail."
                        ),
                    }
                )

        if len(face_ffts) >= 2:
            best_face = max(face_ffts, key=lambda x: float(x.get("fft_score", 0.0)))
            worst_face = min(face_ffts, key=lambda x: float(x.get("fft_score", 0.0)))
            best_s = float(best_face.get("fft_score", 0.0))
            worst_s = float(worst_face.get("fft_score", 0.0))
            if (best_s - worst_s) >= 0.15:
                findings.append(
                    {
                        "title": "Difference Between Faces in Frame",
                        "severity": "warning",
                        "detail": (
                            f"{best_face['global_face_id']} ({self._pct(best_s)}) shows stronger frequency irregularity "
                            f"than {worst_face['global_face_id']} ({self._pct(worst_s)})."
                        ),
                    }
                )

        user_guidance = (
            "How to read the frequency images: Bright straight lines crossing the center come from normal edges "
            "like borders or text boxes. Suspicious AI traces look like matching pairs of bright dots away from the center lines, "
            "or the orange curve bending sharply away from the dotted blue camera trend line."
        )

        return {
            "headline": headline,
            "severity": suspicion,
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_frame(self, frame_m3: Dict[str, Any]) -> Dict[str, Any]:
        noise_interp = self.interpret_noise(frame_m3.get("noise", {}) or {})
        ela_interp = self.interpret_ela(frame_m3.get("ela", {}) or {})
        fft_interp = self.interpret_fft(frame_m3)

        summary = frame_m3.get("summary", {}) or {}
        comb_score = float(summary.get("combined_module3_score", 0.0))
        comb_sev = str(
            summary.get("combined_suspicion", self._severity_label(comb_score))
        ).lower()
        level_label = self._friendly_level(comb_sev)
        pct = self._pct(comb_score)
        ela_used = bool(summary.get("ela_used_in_score", False))

        if comb_sev == "high":
            headline = (
                f"{level_label} ({pct}) — Strong signal-level anomalies (noise, frequency, or compression) "
                "were detected in this frame."
            )
        elif comb_sev == "medium":
            headline = (
                f"{level_label} ({pct}) — Noticeable signal irregularities were found in this frame. "
                "Inspect the Noise & Boundary and 2D-FFT tabs for visual clues."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Signal-level checks look normal in this frame "
                "(note: modern AI fakes can still require Module 4 Deep Learning to detect)."
            )

        key_takeaways = [
            f"Noise & Boundary Check: {noise_interp['headline']}",
            f"Frequency Check (2D-FFT): {fft_interp['headline']}",
        ]
        if ela_used:
            key_takeaways.append(f"JPEG Compression Check (ELA): {ela_interp['headline']}")

        return {
            "headline": headline,
            "severity": comb_sev,
            "key_takeaways": key_takeaways,
            "noise": noise_interp,
            "ela": ela_interp,
            "fft": fft_interp,
        }

    def interpret_manifest(self, m3_manifest: Dict[str, Any]) -> Dict[str, Any]:
        frames = m3_manifest.get("frames_detail", []) or []
        m3_sum = m3_manifest.get("execution_summary", {}) or {}

        if not frames:
            return {
                "headline": "No frames were available for Module 3 analysis.",
                "severity": "low",
                "findings": [],
            }

        max_noise_frame = max(frames, key=lambda f: float(f["summary"]["noise_score"]))
        max_fft_frame = max(frames, key=lambda f: float(f["summary"]["effective_fft_score"]))
        max_comb_frame = max(
            frames, key=lambda f: float(f["summary"]["combined_module3_score"])
        )

        high_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "high")
        med_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "medium")
        low_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "low")

        overall_sev = str(m3_sum.get("overall_suspicion", "low")).lower()
        max_comb = float(m3_sum.get("max_module3_score", 0.0))
        mean_comb = float(m3_sum.get("mean_module3_score", 0.0))
        level_label = self._friendly_level(overall_sev)
        peak_pct = self._pct(max_comb)

        if overall_sev == "high":
            headline = (
                f"{level_label} ({peak_pct}) — Signal forensics flagged strong noise, blending, or frequency anomalies "
                "across the analyzed evidence."
            )
        elif overall_sev == "medium":
            headline = (
                f"{level_label} ({peak_pct}) — Moderate noise, blending, or frequency irregularities were detected. "
                "Combine this evidence with AI deepfake detection (Module 4)."
            )
        else:
            headline = (
                f"{level_label} ({peak_pct}) — No strong signal-level anomalies stood out, though modern AI-generated "
                "media after compression often requires Module 4 (Deep Learning) to detect."
            )

        findings: List[Dict[str, str]] = []
        if len(frames) > 1:
            findings.append(
                {
                    "title": "Frame-by-Frame Breakdown",
                    "severity": "info",
                    "detail": (
                        f"Checked {len(frames)} frames: {high_cnt} High, {med_cnt} Moderate, and {low_cnt} Low suspicion "
                        f"(average across frames: {self._pct(mean_comb)})."
                    ),
                }
            )
            findings.append(
                {
                    "title": "Most Notable Frame(s)",
                    "severity": "warning" if overall_sev in {"medium", "high"} else "ok",
                    "detail": (
                        f"Peak Noise/Boundary score ({self._pct(max_noise_frame['summary']['noise_score'])}) was in Frame #{max_noise_frame['frame_index']} "
                        f"({Path(max_noise_frame['frame_path']).name}); peak Frequency score ({self._pct(max_fft_frame['summary']['effective_fft_score'])}) "
                        f"was in Frame #{max_fft_frame['frame_index']} ({Path(max_fft_frame['frame_path']).name}); "
                        f"overall peak was Frame #{max_comb_frame['frame_index']}."
                    ),
                }
            )
        else:
            s0 = frames[0]["summary"]
            ela_note = (
                f", and JPEG ELA scored {self._pct(s0['ela_score'])}"
                if s0.get("ela_used_in_score")
                else " (JPEG ELA was skipped because source is not native JPEG)"
            )
            findings.append(
                {
                    "title": "Single-Image Breakdown",
                    "severity": "info",
                    "detail": (
                        f"Noise & Boundary scored {self._pct(s0['noise_score'])}, "
                        f"Frequency (2D-FFT) scored {self._pct(s0['effective_fft_score'])}{ela_note}."
                    ),
                }
            )

        findings.append(
            {
                "title": "Forensic Scope Reminder",
                "severity": "info",
                "detail": (
                    "Module 3 checks physical signals (sensor noise mismatches, blending borders, frequency roll-off, and native JPEG compression). "
                    "Absence of these traces does NOT guarantee an image or video is real—always pair Module 3 with Module 4 (Deep Learning) in Module 5."
                ),
            }
        )

        return {
            "headline": headline,
            "severity": overall_sev,
            "findings": findings,
        }
