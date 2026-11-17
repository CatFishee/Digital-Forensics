from pathlib import Path
from typing import Any, Dict, List


class Module3ForensicTranslator:
    """
    Deterministic Rule-Based Forensic Translator for Module 3 (ELA & 2D-FFT).
    Converts quantitative pixel- and frequency-domain metrics into plain-English
    explanations for non-expert investigators without using LLMs.
    """

    @staticmethod
    def _severity_label(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def interpret_ela(self, ela_res: Dict[str, Any]) -> Dict[str, Any]:
        ela_score = float(ela_res.get("ela_score", 0.0))
        suspicion = str(ela_res.get("suspicion_level", self._severity_label(ela_score))).lower()
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

        # 1. Overall ELA verdict headline
        if suspicion == "high":
            headline = (
                f"High compression inconsistency detected (ELA Score: {ela_score:.2f}). "
                "Different regions react very differently to JPEG re-compression, which can indicate spliced/edited areas or heavy graphic overlays."
            )
        elif suspicion == "medium":
            headline = (
                f"Moderate compression inconsistency detected (ELA Score: {ela_score:.2f}). "
                "Some localized differences exist between regions, warranting visual inspection."
            )
        else:
            headline = (
                f"Low compression inconsistency (ELA Score: {ela_score:.2f}). "
                "Pixel compression errors appear relatively uniform across the frame."
            )

        # 2. Input format reliability rule
        if is_jpeg:
            findings.append(
                {
                    "title": "Native JPEG Input",
                    "severity": "ok",
                    "detail": (
                        "The source frame is a native JPEG image, which preserves 8x8 DCT block "
                        "compression history and makes ELA measurements reliable."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": f"Non-JPEG Source Format ({orig_fmt})",
                    "severity": "info",
                    "detail": (
                        f"This frame was analyzed as {orig_fmt} (e.g., an extracted video I-frame). "
                        "Because it is not a native JPEG, ELA reflects re-compression sensitivity rather than "
                        "true double-JPEG history, so moderate scores should be interpreted cautiously."
                    ),
                }
            )

        # 3. Face vs. Background comparison rule
        if face_contrast is not None and face_mean is not None and bg_mean is not None:
            face_val = float(face_mean)
            bg_val = float(bg_mean)
            contrast_val = float(face_contrast)

            if face_val > bg_val:
                direction = "higher error than"
                interpretation = (
                    "This can happen when a sharper or differently compressed face is composited onto a smoother background, "
                    "or when facial features/hair contain dense high-contrast detail."
                )
            else:
                direction = "lower error than"
                interpretation = (
                    "This can occur when an over-smoothed or heavily compressed face sits inside a crisper background scene."
                )

            if contrast_val >= 0.50:
                findings.append(
                    {
                        "title": "Strong Face-vs-Background Compression Mismatch",
                        "severity": "alert",
                        "detail": (
                            f"The detected face region(s) show {direction} the surrounding background "
                            f"(Face Mean: {face_val:.2f} vs. Background Mean: {bg_val:.2f}, Contrast: {contrast_val:.2f}). "
                            f"{interpretation}"
                        ),
                    }
                )
            elif contrast_val >= 0.20:
                findings.append(
                    {
                        "title": "Moderate Face-vs-Background Difference",
                        "severity": "warning",
                        "detail": (
                            f"The detected face region(s) show {direction} the background "
                            f"(Face Mean: {face_val:.2f} vs. Background Mean: {bg_val:.2f}, Contrast: {contrast_val:.2f}). "
                            f"{interpretation}"
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Consistent Face and Background Compression",
                        "severity": "ok",
                        "detail": (
                            f"The face region(s) and surrounding background react similarly to re-compression "
                            f"(Face Mean: {face_val:.2f} vs. Background Mean: {bg_val:.2f}, Contrast: {contrast_val:.2f})."
                        ),
                    }
                )
        else:
            findings.append(
                {
                    "title": "No Face Mask Comparison",
                    "severity": "info",
                    "detail": (
                        "No valid face bounding boxes were available for this frame, so ELA was scored purely "
                        "from global block-level variation."
                    ),
                }
            )

        # 4. Block-grid inconsistency rule
        if p95_p50 >= 2.5 or block_cv >= 0.85:
            findings.append(
                {
                    "title": "Uneven Block-Level Compression Across Frame",
                    "severity": "warning",
                    "detail": (
                        f"The top 5% most reactive 16x16 blocks have {p95_p50:.2f}x the error of median blocks "
                        f"(Block CV: {block_cv:.2f}). This points to localized high-error regions such as edited areas, "
                        "sharp graphic overlays, or added text."
                    ),
                }
            )
        elif p95_p50 >= 1.6 or block_cv >= 0.55:
            findings.append(
                {
                    "title": "Mild Spatial Error Variation",
                    "severity": "info",
                    "detail": (
                        f"Some 16x16 regions differ moderately from the frame median "
                        f"(P95/P50 Ratio: {p95_p50:.2f}x, Block CV: {block_cv:.2f})."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "Uniform Block-Grid Behavior",
                    "severity": "ok",
                    "detail": (
                        f"16x16 image blocks show steady error levels across the frame "
                        f"(P95/P50 Ratio: {p95_p50:.2f}x, Block CV: {block_cv:.2f})."
                    ),
                }
            )

        # 5. Hotspot warning for non-experts
        if hotspot_ratio >= 0.04:
            findings.append(
                {
                    "title": "Concentrated Bright Error Hotspots",
                    "severity": "warning",
                    "detail": (
                        f"{hotspot_ratio * 100:.1f}% of pixels stand out as statistical hotspots. "
                        "Note: Sharp edges (text banners, logos, hairlines) naturally glow bright in ELA—focus on "
                        "whether smooth skin areas look noticeably brighter or darker than smooth background regions."
                    ),
                }
            )

        user_guidance = (
            "How to read ELA images: Bright areas show pixels that changed most during re-saving. "
            "Sharp text, watermarks, and hard borders naturally glow bright; look for unexpected brightness "
            "or smoothness differences inside flat facial skin compared to flat background areas."
        )

        return {
            "headline": headline,
            "severity": suspicion,
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
        suspicion = str(summary.get("fft_suspicion", self._severity_label(effective_score))).lower()

        findings: List[Dict[str, str]] = []

        # 1. Headline
        if suspicion == "high":
            headline = (
                f"Strong frequency-domain anomalies detected (Effective FFT Score: {effective_score:.2f}). "
                "The spectrum contains unusual periodic spikes or elevated high-frequency energy typical of synthetic upsampling."
            )
        elif suspicion == "medium":
            headline = (
                f"Moderate frequency-domain irregularities detected (Effective FFT Score: {effective_score:.2f}). "
                "Mild spectral spikes or high-frequency deviations are present."
            )
        else:
            headline = (
                f"Normal frequency-domain profile (Effective FFT Score: {effective_score:.2f}). "
                "No strong global or face-level periodic AI checkerboard artifacts were detected."
            )

        # Pick the primary target with highest FFT score to inspect detailed spectral metrics
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
        roughness = float(pm.get("radial_roughness", 0.0))

        # 2. Periodic checkerboard spike rule
        if top20_z >= 5.5 or max_z >= 7.5:
            findings.append(
                {
                    "title": f"Periodic Frequency Spikes Detected ({primary_label})",
                    "severity": "alert",
                    "detail": (
                        f"Sharp repeating spikes stand out in the 2D-FFT spectrum "
                        f"(Top-20 Spike Mean Z: {top20_z:.2f}, Max Z: {max_z:.2f}, Density: {spike_density:.5f}). "
                        "Repeating spectral dots away from the center axes are a common fingerprint of AI/GAN upsampling."
                    ),
                }
            )
        elif top20_z >= 3.8 or spike_density >= 0.002:
            findings.append(
                {
                    "title": f"Mild Spectral Spikes ({primary_label})",
                    "severity": "warning",
                    "detail": (
                        f"Minor localized spikes appear in the spectrum "
                        f"(Top-20 Spike Mean Z: {top20_z:.2f}, Max Z: {max_z:.2f}). "
                        "This can reflect mild upsampling noise or crisp repeating patterns."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "No Significant Periodic Checkerboard Spikes",
                    "severity": "ok",
                    "detail": (
                        f"No abnormal repeating peaks were found outside the center DC axes "
                        f"(Top-20 Spike Mean Z: {top20_z:.2f}, Max Z: {max_z:.2f}). "
                        "Note: Heavy JPEG/video compression can smooth out faint synthetic frequency traces."
                    ),
                }
            )

        # 3. High-frequency energy & radial curve decay rule
        if hf_ratio >= 0.72 or roughness >= 0.030:
            findings.append(
                {
                    "title": f"Elevated High-Frequency Energy or Jagged Roll-Off ({primary_label})",
                    "severity": "warning",
                    "detail": (
                        f"High-frequency energy ratio is {hf_ratio:.3f} with radial curve roughness {roughness:.4f}. "
                        "Natural images typically decay smoothly toward the right of the 1D radial plot; elevated or jagged tails "
                        "can indicate artificial sharpening, synthetic textures, or resampling."
                    ),
                }
            )
        elif hf_ratio >= 0.58:
            findings.append(
                {
                    "title": f"Slightly Elevated High-Frequency Tail ({primary_label})",
                    "severity": "info",
                    "detail": (
                        f"High-frequency energy ratio is moderately elevated ({hf_ratio:.3f}), "
                        f"while radial decay roughness remains {roughness:.4f}."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "Smooth Natural Frequency Roll-Off",
                    "severity": "ok",
                    "detail": (
                        f"The 1D radial power curve drops smoothly from low to high frequencies "
                        f"(HF Ratio: {hf_ratio:.3f}, Roughness: {roughness:.4f})."
                    ),
                }
            )

        # 4. Full-frame vs. Unwarped Face ROI comparison rule
        if max_face_score is not None:
            face_val = float(max_face_score)
            if face_val >= frame_score + 0.12:
                findings.append(
                    {
                        "title": "Frequency Anomaly Concentrated Inside Face ROI",
                        "severity": "warning",
                        "detail": (
                            f"The isolated unwarped face crop scores higher ({face_val:.3f}) than the full frame ({frame_score:.3f}), "
                            "indicating that spectral irregularities originate inside the face region rather than the background."
                        ),
                    }
                )
            elif frame_score >= face_val + 0.12:
                findings.append(
                    {
                        "title": "Full-Frame Layout Edges Exceed Face ROI Score",
                        "severity": "info",
                        "detail": (
                            f"The full frame scores higher ({frame_score:.3f}) than the isolated face crop ({face_val:.3f}). "
                            "Straight graphic borders, split-screen dividers, or overlaid banners create bright lines in the "
                            "full-frame 2D-FFT that do not reflect the face itself—check the isolated face spectrum below."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Consistent Full-Frame and Face-ROI Spectra",
                        "severity": "info",
                        "detail": (
                            f"Full-frame FFT ({frame_score:.3f}) and peak unwarped face FFT ({face_val:.3f}) "
                            "show consistent frequency behavior."
                        ),
                    }
                )

        # 5. Multi-face comparison rule (e.g., BEFORE vs. AFTER or two people in scene)
        if len(face_ffts) >= 2:
            best_face = max(face_ffts, key=lambda x: float(x.get("fft_score", 0.0)))
            worst_face = min(face_ffts, key=lambda x: float(x.get("fft_score", 0.0)))
            best_s = float(best_face.get("fft_score", 0.0))
            worst_s = float(worst_face.get("fft_score", 0.0))

            if (best_s - worst_s) >= 0.12:
                findings.append(
                    {
                        "title": "Difference Between Detected Faces",
                        "severity": "warning",
                        "detail": (
                            f"{best_face['global_face_id']} shows a higher frequency anomaly ({best_s:.3f}) "
                            f"than {worst_face['global_face_id']} ({worst_s:.3f}), suggesting spectral differences "
                            "are localized to a specific face."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Similar Frequency Profiles Across Faces",
                        "severity": "info",
                        "detail": (
                            f"All {len(face_ffts)} detected faces share comparable FFT scores "
                            f"(range: {worst_s:.3f} to {best_s:.3f})."
                        ),
                    }
                )

        user_guidance = (
            "How to read 2D-FFT: Bright horizontal, vertical, or tilted straight lines crossing the center come from "
            "image borders, dividers, or text boxes. Suspect AI artifacts look like symmetric bright dots/grid patterns "
            "away from the center lines, or bumpy spikes past the dashed line on the 1D curve."
        )

        return {
            "headline": headline,
            "severity": suspicion,
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_frame(self, frame_m3: Dict[str, Any]) -> Dict[str, Any]:
        ela_interp = self.interpret_ela(frame_m3.get("ela", {}) or {})
        fft_interp = self.interpret_fft(frame_m3)
        summary = frame_m3.get("summary", {}) or {}
        comb_score = float(summary.get("combined_module3_score", 0.0))
        comb_sev = str(summary.get("combined_suspicion", self._severity_label(comb_score))).lower()

        if comb_sev == "high":
            headline = (
                f"High pixel/frequency suspicion in this frame (Combined Score: {comb_score:.2f}). "
                "Significant compression or spectral anomalies were flagged."
            )
        elif comb_sev == "medium":
            headline = (
                f"Moderate pixel/frequency suspicion in this frame (Combined Score: {comb_score:.2f}). "
                "Review the ELA and isolated Face-ROI FFT findings before concluding."
            )
        else:
            headline = (
                f"Low pixel/frequency suspicion in this frame (Combined Score: {comb_score:.2f}). "
                "Traditional compression and spectral checks did not find strong tampering signals."
            )

        key_takeaways = [ela_interp["headline"], fft_interp["headline"]]
        return {
            "headline": headline,
            "severity": comb_sev,
            "key_takeaways": key_takeaways,
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

        max_ela_frame = max(frames, key=lambda f: float(f["summary"]["ela_score"]))
        max_fft_frame = max(frames, key=lambda f: float(f["summary"]["effective_fft_score"]))
        max_comb_frame = max(frames, key=lambda f: float(f["summary"]["combined_module3_score"]))

        high_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "high")
        med_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "medium")
        low_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "low")

        overall_sev = str(m3_sum.get("overall_suspicion", "low")).lower()
        max_comb = float(m3_sum.get("max_module3_score", 0.0))
        mean_comb = float(m3_sum.get("mean_module3_score", 0.0))

        if overall_sev == "high":
            headline = (
                f"Module 3 detected strong pixel or frequency anomalies (Peak Combined Score: {max_comb:.2f}, "
                f"Mean: {mean_comb:.2f})."
            )
        elif overall_sev == "medium":
            headline = (
                f"Module 3 detected moderate pixel or frequency irregularities (Peak Combined Score: {max_comb:.2f}, "
                f"Mean: {mean_comb:.2f})."
            )
        else:
            headline = (
                f"Module 3 found low overall pixel/frequency suspicion (Peak Combined Score: {max_comb:.2f}, "
                f"Mean: {mean_comb:.2f})."
            )

        findings: List[Dict[str, str]] = [
            {
                "title": "Frame-by-Frame Distribution",
                "severity": "info",
                "detail": (
                    f"Out of {len(frames)} analyzed frame(s): {high_cnt} High, {med_cnt} Medium, and {low_cnt} Low suspicion."
                ),
            },
            {
                "title": "Peak Anomaly Locations",
                "severity": "warning" if overall_sev in {"medium", "high"} else "ok",
                "detail": (
                    f"Highest ELA score ({max_ela_frame['summary']['ela_score']:.3f}) occurred at Frame #{max_ela_frame['frame_index']} "
                    f"({Path(max_ela_frame['frame_path']).name}); highest FFT score ({max_fft_frame['summary']['effective_fft_score']:.3f}) "
                    f"occurred at Frame #{max_fft_frame['frame_index']} ({Path(max_fft_frame['frame_path']).name}); "
                    f"peak combined frame is Frame #{max_comb_frame['frame_index']}."
                ),
            },
            {
                "title": "Forensic Scope Reminder",
                "severity": "info",
                "detail": (
                    "Module 3 measures traditional compression (ELA) and frequency (2D-FFT) clues only. "
                    "Heavy social-media compression can mask these clues, while text/graphic overlays can raise ELA/FFT scores. "
                    "Always combine these findings with Module 4 (Deep Learning + Grad-CAM) in Module 5."
                ),
            },
        ]

        return {
            "headline": headline,
            "severity": overall_sev,
            "findings": findings,
        }
