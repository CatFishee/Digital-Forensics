from pathlib import Path
from typing import Any, Dict, List


class Module3ForensicTranslator:
    """
    Deterministic Rule-Based Forensic Translator for Module 3:
      - Noise Residual & Intra-Face Patch Consistency (SRM)
      - Face-Boundary Blending & Internal Splice Seam Analysis
      - Temporal Face Flicker & Inter-Frame Consistency (Video Bursts)
      - 2D-FFT Spectral, Quadrant Discrepancy & 1/f Decay Analysis
    All metric keys match the analyzers 1:1 and use simple plain-English phrasing.
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

        intra_cv = metrics.get("intra_face_noise_cv")
        intra_p90_p10 = metrics.get("intra_face_p90_p10_ratio")
        intra_outliers = metrics.get("intra_face_outlier_ratio")

        findings: List[Dict[str, str]] = []

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Strong pixel-grain mismatches were found. "
                "Grain levels vary unnaturally across the face or differ sharply from the surrounding scene, "
                "which commonly occurs with face-swaps, inpainting, or localized retouching."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate pixel-grain differences appear in this frame. "
                "Different parts of the face show conflicting grain intensity."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Pixel grain looks consistent across the frame. "
                "No strong noise-floor mismatches stood out."
            )

        # 1. Intra-face patch noise dispersion (catches localized component splices / inpainting)
        if intra_cv is not None and intra_p90_p10 is not None:
            c_val = float(intra_cv)
            p_val = float(intra_p90_p10)
            if c_val >= 1.20 or p_val >= 4.20:
                findings.append(
                    {
                        "title": "Severe Grain Discrepancy Across Face Patches",
                        "severity": "alert",
                        "detail": (
                            "Different regions of the same face exhibit sharply conflicting camera grain. "
                            "One facial patch is significantly smoother or grainier than adjacent skin, "
                            "a signature of localized component splicing or inpainting."
                        ),
                    }
                )
            elif c_val >= 0.85 or p_val >= 2.80:
                findings.append(
                    {
                        "title": "Uneven Grain Variation Across Face Skin",
                        "severity": "warning",
                        "detail": (
                            "Skin texture grain varies noticeably between different areas of the face, "
                            "which can indicate localized retouching or partial face edits."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Uniform Facial Skin Grain",
                        "severity": "ok",
                        "detail": "Pixel grain distribution remains consistent across all regions of the face.",
                    }
                )

        if intra_outliers is not None and float(intra_outliers) >= 0.18:
            findings.append(
                {
                    "title": "Localized Noise Outlier Cluster",
                    "severity": "warning",
                    "detail": (
                        "A distinct cluster of blocks inside the face deviates sharply from the dominant "
                        "facial noise floor, pointing to a localized pasted region."
                    ),
                }
            )

        # 2. Global Face vs Background noise contrast
        if face_contrast is not None and face_noise is not None and bg_noise is not None:
            fc_val = float(face_contrast)
            direction_text = (
                "smoother (less natural grain) than the background"
                if face_smoother
                else "noisier or sharper than the background"
            )
            if fc_val >= 0.90:
                findings.append(
                    {
                        "title": "Strong Grain Mismatch Between Face and Scene",
                        "severity": "alert",
                        "detail": (
                            f"The face skin looks noticeably {direction_text}. "
                            "Real camera scenes usually share a consistent sensor grain; pasted or AI-generated faces often do not."
                        ),
                    }
                )
            elif fc_val >= 0.45:
                findings.append(
                    {
                        "title": "Moderate Grain Difference Between Face and Scene",
                        "severity": "warning",
                        "detail": (
                            f"The face skin appears somewhat {direction_text}. "
                            "This can happen with face-swaps, beauty filters, or strong background blur."
                        ),
                    }
                )

        if chroma_contrast is not None and float(chroma_contrast) >= 0.60:
            findings.append(
                {
                    "title": "Color-Grain Mismatch Inside Face",
                    "severity": "warning",
                    "detail": (
                        "Fine color grain inside the face differs noticeably from the rest of the frame, "
                        "which can occur when a face comes from a different camera or AI generator."
                    ),
                }
            )

        if p95_p50 >= 6.0 or block_cv >= 1.35:
            findings.append(
                {
                    "title": "Uneven Grain Patches Across Background",
                    "severity": "warning",
                    "detail": (
                        "Some regions of the frame have noticeably heavier or smoother grain than others, "
                        "which can point to pasted regions or selective retouching."
                    ),
                }
            )

        user_guidance = (
            "How to read the images above: The left image isolates fine pixel grain (SRM residual); "
            "the heatmap on the right displays smooth/low-grain areas in dark blue/purple and high-grain areas in bright yellow-red."
        )

        return {
            "headline": headline,
            "severity": suspicion,
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_boundary(self, boundary_res: Dict[str, Any]) -> Dict[str, Any]:
        score = float(boundary_res.get("boundary_score", 0.0))
        suspicion = str(
            boundary_res.get("suspicion_level", self._severity_label(score))
        ).lower()
        level_label = self._friendly_level(suspicion)
        pct = self._pct(score)
        metrics = boundary_res.get("metrics", {}) or {}

        evaluated = int(metrics.get("faces_evaluated", 0) or 0)
        seam_halo = metrics.get("seam_halo_ratio")
        chroma_diff = metrics.get("skin_context_chroma_diff")
        chroma_bleed = metrics.get("seam_chroma_bleed")
        sharp_ratio = metrics.get("inner_outer_sharpness_ratio")
        internal_seam = metrics.get("internal_seam_ratio")
        bilateral_chroma = metrics.get("bilateral_chroma_diff")
        bilateral_grad = metrics.get("bilateral_gradient_ratio")
        unreliable = bool(metrics.get("unreliable_face_warning", False))
        asym_detected = bool(metrics.get("anatomical_asymmetry_detected", False))

        findings: List[Dict[str, str]] = []

        if evaluated == 0:
            return {
                "headline": "Not Applicable — No face was detected to inspect blending borders.",
                "severity": "low",
                "findings": [
                    {
                        "title": "No Face Available",
                        "severity": "info",
                        "detail": "Face-boundary blending checks require at least one detected face.",
                    }
                ],
                "user_guidance": "Upload a frame containing a clearly visible face to inspect blending borders and color matching.",
            }

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Clear blending seams, internal inpainting edges, or skin-tone mismatches "
                "were detected. The cheeks/jawline or internal facial features do not blend naturally."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate blending or color differences appear along the face border or midface. "
                "Inspect the heatmap for unnatural seams or bilateral tone shifts."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — The face edge, internal features, and skin tone blend naturally."
            )

        if asym_detected:
            findings.append(
                {
                    "title": "Severe Bilateral Facial Asymmetry Detected",
                    "severity": "alert",
                    "detail": (
                        "The relative alignment of the eyes and nose deviates sharply from natural bilateral symmetry "
                        "on this frontal face, characteristic of spliced facial components (e.g. mismatched eye/nose sources)."
                    ),
                }
            )

        if unreliable:
            findings.append(
                {
                    "title": "Edge-Clipped Face Warning",
                    "severity": "warning",
                    "detail": (
                        "The face is partially cut off at the image boundary. Outer perimeter border metrics "
                        "may be influenced by frame borders."
                    ),
                }
            )

        # Internal splice seam findings (bridges the visual heatmap to the verbal verdict)
        if internal_seam is not None:
            is_val = float(internal_seam)
            if is_val >= 3.20:
                findings.append(
                    {
                        "title": "Strong Internal Splice or Inpainting Seam",
                        "severity": "alert",
                        "detail": (
                            "A sharp, unnatural gradient boundary cuts across smooth facial skin "
                            "(e.g. across the nose or cheek), matching the bright seam visible on the heatmap."
                        ),
                    }
                )
            elif is_val >= 2.40:
                findings.append(
                    {
                        "title": "Mild Internal Feature Seam",
                        "severity": "warning",
                        "detail": "A faint gradient step-edge is present across the internal face canvas.",
                    }
                )

        # Bilateral chrominance / lighting symmetry
        if bilateral_chroma is not None and float(bilateral_chroma) >= 12.0:
            findings.append(
                {
                    "title": "Bilateral Skin-Tone Split",
                    "severity": "alert",
                    "detail": (
                        f"The left and right sides of the face exhibit a substantial color tone discrepancy "
                        f"(Lab ΔE: {float(bilateral_chroma):.1f}), indicating composite lighting or mismatched sources."
                    ),
                }
            )
        elif bilateral_chroma is not None and float(bilateral_chroma) >= 8.0:
            findings.append(
                {
                    "title": "Mild Bilateral Color Asymmetry",
                    "severity": "warning",
                    "detail": "Slight color tint discrepancy between the left and right facial halves.",
                }
            )

        # Perimeter jawline/neck checks
        if seam_halo is not None:
            sh_val = float(seam_halo)
            if sh_val >= 2.6:
                findings.append(
                    {
                        "title": "Strong Blending Halo Along Jawline",
                        "severity": "alert",
                        "detail": "An unusually sharp or double-edged transition appears along the lower face outline.",
                    }
                )
            elif sh_val >= 1.95:
                findings.append(
                    {
                        "title": "Mild Edge Transition Along Jawline",
                        "severity": "warning",
                        "detail": "A slight sharpness change appears around the cheeks or jawline border.",
                    }
                )

        if chroma_diff is not None and float(chroma_diff) >= 18.0:
            findings.append(
                {
                    "title": "Strong Face-vs-Neck Color Mismatch",
                    "severity": "alert",
                    "detail": "Skin color inside the face differs noticeably from the outer neck and surroundings.",
                }
            )

        if chroma_bleed is not None and float(chroma_bleed) >= 2.0:
            findings.append(
                {
                    "title": "Color Bleed Along Face Border",
                    "severity": "warning",
                    "detail": "Color fringes appear along the face outline, often left behind by masked blending.",
                }
            )

        if sharp_ratio is not None and float(sharp_ratio) >= 2.5:
            findings.append(
                {
                    "title": "Sharpness Mismatch (Inner Face vs. Outer Scene)",
                    "severity": "warning",
                    "detail": "The inside of the face is noticeably softer or sharper than the immediate surroundings.",
                }
            )

        user_guidance = (
            "How to read the images above: The left image shows the inspected boundary zones (ellipses outline the jawline); "
            "the heatmap on the right glows bright orange/yellow wherever abrupt gradient cuts, color fringes, or inpainting seams occur."
        )

        return {
            "headline": headline,
            "severity": suspicion,
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_temporal(self, temporal_res: Dict[str, Any]) -> Dict[str, Any]:
        if not temporal_res.get("applicable", False):
            return {
                "headline": "Not Applicable (Single Image) — Frame-to-frame flicker checks only run on video files.",
                "severity": "low",
                "findings": [
                    {
                        "title": "Video Only Check",
                        "severity": "info",
                        "detail": (
                            "Because this evidence is a single static image, there are no consecutive video frames "
                            "to check for face flicker or landmark jitter."
                        ),
                    }
                ],
                "user_guidance": "Upload a video file to inspect frame-by-frame face flicker across consecutive frames.",
            }

        score = float(temporal_res.get("temporal_score", 0.0))
        suspicion = str(
            temporal_res.get("suspicion_level", self._severity_label(score))
        ).lower()
        level_label = self._friendly_level(suspicion)
        pct = self._pct(score)
        metrics = temporal_res.get("metrics", {}) or {}

        tracked = int(metrics.get("tracked_frames", 0) or 0)
        lm_jitter = metrics.get("landmark_jitter")
        tex_flicker = metrics.get("texture_flicker_ratio")
        col_flicker = metrics.get("color_flicker_lab")
        lum_flicker = metrics.get("brightness_flicker")
        unreliable = bool(metrics.get("unreliable_face_warning", False))
        faces_tracked = int(metrics.get("faces_tracked_count", 0) or 0)
        peak_face_id = metrics.get("peak_face_id") or "Face"

        per_face_scores = temporal_res.get("faces", []) or []

        findings: List[Dict[str, str]] = []

        if tracked < 2:
            return {
                "headline": "Low Suspicion (0%) — No consistently tracked face was found across this short frame burst.",
                "severity": "low",
                "findings": [
                    {
                        "title": "Face Not Tracked Across Burst",
                        "severity": "info",
                        "detail": "A face could not be followed across at least 2 consecutive frames in this burst.",
                    }
                ],
                "user_guidance": "Check another video frame where the face remains clearly visible across consecutive frames.",
            }

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Noticeable frame-to-frame face flicker or jitter was found. "
                "The face shape, skin brightness, or texture pops unnaturally between consecutive frames."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Mild frame-to-frame face instability was detected across this burst."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Face features, brightness, and skin texture stay steady across consecutive frames."
            )

        if unreliable:
            findings.append(
                {
                    "title": "Edge-Clipped Face Notice",
                    "severity": "warning",
                    "detail": "Face is close to the frame boundary during burst tracking.",
                }
            )

        if faces_tracked >= 2 and per_face_scores:
            parts = [
                f"{pf.get('global_face_id', 'Face')}: {self._pct(pf.get('temporal_score', 0.0))}"
                for pf in per_face_scores
            ]
            findings.append(
                {
                    "title": "All Tracked Faces in This Burst",
                    "severity": "info",
                    "detail": (
                        f"Tracked {faces_tracked} faces independently: {', '.join(parts)}. "
                        f"Highest suspicion was detected on {peak_face_id}."
                    ),
                }
            )

        if lm_jitter is not None and float(lm_jitter) >= 0.025:
            findings.append(
                {
                    "title": "Facial Feature Jitter Detected",
                    "severity": "alert",
                    "detail": "Eyes, nose, or mouth positions wobble noticeably from one frame to the next.",
                }
            )

        if tex_flicker is not None and float(tex_flicker) >= 0.90:
            findings.append(
                {
                    "title": "Skin Texture Flickering",
                    "severity": "warning",
                    "detail": "Fine skin details change rapidly between adjacent frames, typical of per-frame generation.",
                }
            )

        if (lum_flicker is not None and float(lum_flicker) >= 2.8) or (
            col_flicker is not None and float(col_flicker) >= 2.0
        ):
            findings.append(
                {
                    "title": "Brightness or Color Popping",
                    "severity": "warning",
                    "detail": "Skin brightness or color tint fluctuates suddenly across consecutive frames.",
                }
            )

        user_guidance = (
            "How to read the visuals above: The heatmap glows bright where skin texture flickers between consecutive frames; "
            "the stability plots on the right should remain steady in a normal real video."
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
        suspicion = str(
            summary.get("fft_suspicion", self._severity_label(effective_score))
        ).lower()
        level_label = self._friendly_level(suspicion)
        pct = self._pct(effective_score)

        findings: List[Dict[str, str]] = []

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Strong frequency anomalies were detected. "
                "Repeating grid dots, sub-region spectral divergence, or abnormal detail roll-off appear, "
                "characteristic of synthetic generation or face inpainting."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate frequency irregularities were found. "
                "Some sub-region fine-detail patterns deviate from a typical natural camera curve."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Frequency patterns look natural. "
                "No strong repeating grid traces or sub-region spectral mismatches were found."
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
        quad_slope = pm.get("quadrant_slope_discrepancy")
        quad_hf = pm.get("quadrant_hf_discrepancy")

        # Sub-region / quadrant spectral divergence
        if quad_slope is not None and float(quad_slope) >= 0.95:
            findings.append(
                {
                    "title": f"Sub-Region Frequency Discrepancy ({primary_label})",
                    "severity": "alert",
                    "detail": (
                        "Different quadrants of the face exhibit sharply conflicting frequency decay slopes. "
                        "One facial quadrant possesses unnatural smoothness or synthetic high frequencies "
                        "compared to adjacent quadrants."
                    ),
                }
            )
        elif quad_slope is not None and float(quad_slope) >= 0.65:
            findings.append(
                {
                    "title": f"Mild Quadrant Frequency Divergence ({primary_label})",
                    "severity": "warning",
                    "detail": "Minor discrepancy in frequency decay between different quadrants of the face crop.",
                }
            )

        if top20_z >= 5.2 or max_z >= 7.0:
            findings.append(
                {
                    "title": f"Repeating Grid Pattern Found ({primary_label})",
                    "severity": "alert",
                    "detail": (
                        "Unusual repeating bright spikes appear in the frequency scan, "
                        "a hallmark of GAN/diffusion convolutional upsampling."
                    ),
                }
            )
        elif top20_z >= 3.7 or spike_density >= 0.002:
            findings.append(
                {
                    "title": f"Mild Repeating Patterns ({primary_label})",
                    "severity": "warning",
                    "detail": "A few repeating frequency spikes appear outside the normal camera range.",
                }
            )
        else:
            findings.append(
                {
                    "title": "No Repeating AI Grid Spikes",
                    "severity": "ok",
                    "detail": "No sharp repeating checkerboard spikes were detected.",
                }
            )

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
                    "detail": "The radial spectrum curve bends away from a natural camera 1/f decay slope.",
                }
            )
        else:
            findings.append(
                {
                    "title": "Natural Detail Roll-Off",
                    "severity": "ok",
                    "detail": "The frequency curve follows a smooth, natural camera decay from low to high frequency.",
                }
            )

        user_guidance = (
            "How to read the frequency images: Straight cross lines through the center reflect natural image borders; "
            "suspicious synthetic traces appear as symmetrical bright dot pairs off-axis, or as sharp deviations "
            "between the observed orange curve and the blue 1/f camera trend line."
        )

        return {
            "headline": headline,
            "severity": suspicion,
            "findings": findings,
            "user_guidance": user_guidance,
        }

    def interpret_frame(self, frame_m3: Dict[str, Any]) -> Dict[str, Any]:
        boundary_interp = self.interpret_boundary(frame_m3.get("boundary", {}) or {})
        temporal_interp = self.interpret_temporal(frame_m3.get("temporal", {}) or {})
        noise_interp = self.interpret_noise(frame_m3.get("noise", {}) or {})
        fft_interp = self.interpret_fft(frame_m3)

        summary = frame_m3.get("summary", {}) or {}
        comb_score = float(summary.get("combined_module3_score", 0.0))
        comb_sev = str(
            summary.get("combined_suspicion", self._severity_label(comb_score))
        ).lower()
        level_label = self._friendly_level(comb_sev)
        pct = self._pct(comb_score)
        temporal_used = bool(summary.get("temporal_applicable", False))

        if comb_sev == "high":
            headline = (
                f"{level_label} ({pct}) — Strong physical signal anomalies were detected in this frame."
            )
        elif comb_sev == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate signal irregularities were found in this frame. "
                "Inspect the Face Border & Seams, Grain, and Frequency tabs for visual confirmation."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Physical signal checks look mostly normal in this frame."
            )

        key_takeaways = [
            f"Face Border & Internal Seams: {boundary_interp['headline']}",
            f"Pixel Grain (Noise Consistency): {noise_interp['headline']}",
            f"Frequency Check (2D-FFT & Quadrants): {fft_interp['headline']}",
        ]
        if temporal_used:
            key_takeaways.insert(
                1, f"Video Frame Flicker: {temporal_interp['headline']}"
            )

        return {
            "headline": headline,
            "severity": comb_sev,
            "key_takeaways": key_takeaways,
            "boundary": boundary_interp,
            "temporal": temporal_interp,
            "noise": noise_interp,
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
                "peak_frames": [],
            }

        max_comb_frame = max(
            frames, key=lambda f: float(f["summary"]["combined_module3_score"])
        )
        max_bound_frame = max(
            frames, key=lambda f: float(f["summary"]["boundary_score"])
        )
        max_noise_frame = max(
            frames, key=lambda f: float(f["summary"]["noise_score"])
        )
        max_fft_frame = max(
            frames, key=lambda f: float(f["summary"]["effective_fft_score"])
        )

        overall_sev = str(m3_sum.get("overall_suspicion", "low")).lower()
        max_comb = float(m3_sum.get("max_module3_score", 0.0))
        mean_comb = float(m3_sum.get("mean_module3_score", 0.0))
        level_label = self._friendly_level(overall_sev)
        peak_pct = self._pct(max_comb)

        if overall_sev == "high":
            headline = (
                f"{level_label} ({peak_pct}) — Signal forensics flagged strong blending seams, "
                "patch grain inconsistency, or frequency anomalies."
            )
        elif overall_sev == "medium":
            headline = (
                f"{level_label} ({peak_pct}) — Moderate signal irregularities were detected across frames."
            )
        else:
            headline = (
                f"{level_label} ({peak_pct}) — Physical signal checks found no strong anomalies across frames."
            )

        findings: List[Dict[str, str]] = []
        if len(frames) > 1:
            high_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "high")
            med_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "medium")
            low_cnt = sum(1 for f in frames if f["summary"]["combined_suspicion"] == "low")
            findings.append(
                {
                    "title": "Frame-by-Frame Breakdown",
                    "severity": "info",
                    "detail": (
                        f"Checked {len(frames)} frames: {high_cnt} High, {med_cnt} Moderate, and {low_cnt} Low suspicion "
                        f"(average: {self._pct(mean_comb)})."
                    ),
                }
            )

        peak_frames: List[Dict[str, Any]] = [
            {
                "label": "Peak Border & Seams" if len(frames) > 1 else "Border & Seams",
                "frame_index": max_bound_frame["frame_index"],
                "score": float(max_bound_frame["summary"]["boundary_score"]),
                "severity": str(max_bound_frame["summary"]["boundary_suspicion"]).lower(),
                "applicable": bool(max_bound_frame["summary"].get("boundary_evaluated", True)),
            }
        ]
        if m3_sum.get("temporal_analyzed"):
            max_temp_frame = max(
                frames, key=lambda f: float(f["summary"]["temporal_score"])
            )
            peak_frames.append(
                {
                    "label": "Peak Video Flicker" if len(frames) > 1 else "Video Flicker",
                    "frame_index": max_temp_frame["frame_index"],
                    "score": float(max_temp_frame["summary"]["temporal_score"]),
                    "severity": str(max_temp_frame["summary"]["temporal_suspicion"]).lower(),
                    "applicable": True,
                }
            )
        peak_frames.extend(
            [
                {
                    "label": "Peak Pixel Grain" if len(frames) > 1 else "Pixel Grain",
                    "frame_index": max_noise_frame["frame_index"],
                    "score": float(max_noise_frame["summary"]["noise_score"]),
                    "severity": str(max_noise_frame["summary"]["noise_suspicion"]).lower(),
                    "applicable": True,
                },
                {
                    "label": "Peak Frequency" if len(frames) > 1 else "Frequency",
                    "frame_index": max_fft_frame["frame_index"],
                    "score": float(max_fft_frame["summary"]["effective_fft_score"]),
                    "severity": str(max_fft_frame["summary"]["fft_suspicion"]).lower(),
                    "applicable": True,
                },
                {
                    "label": "Overall Peak" if len(frames) > 1 else "Overall",
                    "frame_index": max_comb_frame["frame_index"],
                    "score": float(max_comb_frame["summary"]["combined_module3_score"]),
                    "severity": str(max_comb_frame["summary"]["combined_suspicion"]).lower(),
                    "applicable": True,
                },
            ]
        )

        return {
            "headline": headline,
            "severity": overall_sev,
            "findings": findings,
            "peak_frames": peak_frames,
            "multi_frame": len(frames) > 1,
        }
