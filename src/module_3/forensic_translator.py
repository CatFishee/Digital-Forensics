from pathlib import Path
from typing import Any, Dict, List


class Module3ForensicTranslator:
    """
    Deterministic Rule-Based Forensic Translator for Module 3:
      - Noise Residual Consistency (SRM)
      - Face-Boundary Blending & Chrominance Mismatch
      - Temporal Face Flicker & Inter-Frame Consistency (Video Bursts)
      - 2D-FFT Spectral & 1/f Decay Analysis
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

        findings: List[Dict[str, str]] = []

        if suspicion == "high":
            headline = (
                f"{level_label} ({pct}) — Strong pixel-grain mismatches were found. "
                "The skin texture inside the face differs sharply from the surrounding scene, "
                "which can happen when a face is swapped or edited."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Moderate pixel-grain differences appear in this frame. "
                "The face skin texture behaves somewhat differently from the background."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Pixel grain looks consistent across the frame. "
                "No strong noise-floor mismatches stood out."
            )

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
                        "title": "Strong Grain Mismatch Between Face Skin and Scene",
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
                        "title": "Moderate Grain Difference Between Face Skin and Scene",
                        "severity": "warning",
                        "detail": (
                            f"The face skin appears somewhat {direction_text}. "
                            "This can happen with face-swaps, beauty filters, or strong background blur."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Consistent Face and Background Grain",
                        "severity": "ok",
                        "detail": "The fine pixel grain on the face skin matches the surrounding scene well.",
                    }
                )
        else:
            findings.append(
                {
                    "title": "No Face Detected for Grain Comparison",
                    "severity": "info",
                    "detail": "No valid face box was available, so only overall frame grain uniformity was checked.",
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
                    "title": "Uneven Grain Patches Across Frame",
                    "severity": "warning",
                    "detail": (
                        "Some regions of the frame have noticeably heavier or smoother grain than others, "
                        "which can point to pasted regions or selective retouching."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "title": "Steady Background Grain",
                    "severity": "ok",
                    "detail": "Grain levels across regions stay within a normal natural range.",
                }
            )

        user_guidance = (
            "How to read the images above: The left image hides colors and lighting to show only tiny pixel grain; "
            "the heatmap on the right shows smooth/low-grain areas in dark blue and high-grain areas in yellow-red."
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
                f"{level_label} ({pct}) — Clear blending or color mismatches appear around the edge of the face. "
                "The cheeks/jawline border or skin color tone does not transition naturally into the neck and surroundings."
            )
        elif suspicion == "medium":
            headline = (
                f"{level_label} ({pct}) — Mild blending or color differences appear near the face border. "
                "Inspect the face outline in the heatmap for faint seams or tone shifts."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — The face edge and skin tone blend naturally into the neck and surroundings."
            )

        if seam_halo is not None:
            sh_val = float(seam_halo)
            if sh_val >= 2.6:
                findings.append(
                    {
                        "title": "Strong Blending Halo Along Face Edge",
                        "severity": "alert",
                        "detail": "An unusually sharp or double-edged transition appears along the lower face outline.",
                    }
                )
            elif sh_val >= 1.95:
                findings.append(
                    {
                        "title": "Mild Edge Transition Along Face Outline",
                        "severity": "warning",
                        "detail": "A slight sharpness change appears around the cheeks or jawline border.",
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Clean Face-to-Neck Edge",
                        "severity": "ok",
                        "detail": "No artificial blending halo was detected along the jawline and cheeks.",
                    }
                )

        if chroma_diff is not None:
            cd_val = float(chroma_diff)
            if cd_val >= 18.0:
                findings.append(
                    {
                        "title": "Strong Skin-Color Mismatch",
                        "severity": "alert",
                        "detail": (
                            "The color tone inside the face differs noticeably from the immediate neck or outer jaw area, "
                            "which is common when a swapped face is imperfectly color-matched."
                        ),
                    }
                )
            elif cd_val >= 11.5:
                findings.append(
                    {
                        "title": "Slight Skin-Tone Shift",
                        "severity": "warning",
                        "detail": "The face tint shifts slightly compared to the outer neck/jaw area.",
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Consistent Skin Color Tone",
                        "severity": "ok",
                        "detail": "Face skin color matches the immediate neck and surroundings naturally.",
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
                    "detail": (
                        "The inside of the face is noticeably softer or sharper than the immediate surroundings, "
                        "which can happen when a lower-resolution swapped face is pasted onto a sharper scene."
                    ),
                }
            )

        user_guidance = (
            "How to read the images above: The left image shows the exact jawline/cheek border zone being checked; "
            "the heatmap on the right glows bright orange/yellow if there are color fringes or unnatural seams along that border."
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

        if lm_jitter is not None:
            lj = float(lm_jitter)
            if lj >= 0.025:
                findings.append(
                    {
                        "title": "Facial Feature Jitter Detected",
                        "severity": "alert",
                        "detail": "Eyes, nose, or mouth positions wobble noticeably from one frame to the next.",
                    }
                )
            elif lj >= 0.015:
                findings.append(
                    {
                        "title": "Slight Feature Wobble",
                        "severity": "warning",
                        "detail": "Minor frame-to-frame wobble appears in facial landmark alignment.",
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Steady Facial Alignment",
                        "severity": "ok",
                        "detail": "Eyes, nose, and mouth move smoothly from frame to frame without artificial jitter.",
                    }
                )

        if tex_flicker is not None and float(tex_flicker) >= 0.90:
            findings.append(
                {
                    "title": "Skin Texture Flickering",
                    "severity": "warning",
                    "detail": (
                        "Fine skin details change rapidly between adjacent frames even after aligning the face, "
                        "which can happen when a deepfake generates each frame independently."
                    ),
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
        else:
            findings.append(
                {
                    "title": "Consistent Skin Color & Lighting",
                    "severity": "ok",
                    "detail": "Face brightness and color tone remain smooth across the burst.",
                }
            )

        user_guidance = (
            "How to read the visuals above: The heatmap glows bright where skin texture flickers between consecutive frames; "
            "the charts on the right should stay flat and below the dashed grey line in a normal real video."
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
                "Unusual repeating grid dots or abnormal detail roll-off appear, "
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
                    "detail": "A few small repeating spots appear in the frequency scan.",
                }
            )
        else:
            findings.append(
                {
                    "title": "No Repeating AI Grid Dots Found",
                    "severity": "ok",
                    "detail": (
                        "No sharp repeating grid dots stood out. Note that resizing or compression "
                        "can smooth out faint grid dots."
                    ),
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
                    "detail": (
                        "The orange spectrum curve bends away from a normal natural camera slope, "
                        "which can point to synthetic textures, upscaling, or over-smoothing."
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

        if max_face_score is not None:
            face_val = float(max_face_score)
            if face_val >= frame_score + 0.15:
                findings.append(
                    {
                        "title": "Anomaly Concentrated Inside Face",
                        "severity": "warning",
                        "detail": (
                            f"The zoomed-in face scan ({self._pct(face_val)}) scores higher than the full frame ({self._pct(frame_score)})."
                        ),
                    }
                )
            elif frame_score >= face_val + 0.15:
                findings.append(
                    {
                        "title": "Full Frame Higher Than Face",
                        "severity": "info",
                        "detail": (
                            f"The full frame ({self._pct(frame_score)}) scores higher than the isolated face ({self._pct(face_val)})."
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
                "Inspect the Face Border, Flicker, Grain, and 2D-FFT tabs for visual clues."
            )
        else:
            headline = (
                f"{level_label} ({pct}) — Physical signal checks look mostly normal in this frame "
                "(note: modern AI fakes can still require Module 4 Deep Learning to detect)."
            )

        key_takeaways = [
            f"Face Border & Color Match: {boundary_interp['headline']}",
            f"Pixel Grain (Noise): {noise_interp['headline']}",
            f"Frequency Check (2D-FFT): {fft_interp['headline']}",
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
                f"{level_label} ({peak_pct}) — Signal forensics flagged strong blending, flicker, grain, or frequency anomalies."
            )
        elif overall_sev == "medium":
            headline = (
                f"{level_label} ({peak_pct}) — Moderate signal irregularities were detected. "
                "Combine this evidence with AI deepfake detection (Module 4)."
            )
        else:
            headline = (
                f"{level_label} ({peak_pct}) — No strong physical signal anomalies stood out, though modern AI-generated "
                "media often requires Module 4 (Deep Learning) to detect."
            )

        findings: List[Dict[str, str]] = []
        if len(frames) > 1:
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
                "label": "Peak Border & Color" if len(frames) > 1 else "Border & Color",
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
