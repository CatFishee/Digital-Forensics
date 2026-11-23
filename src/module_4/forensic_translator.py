from __future__ import annotations

from typing import Any, Dict, List


class Module4ForensicTranslator:
    """
    Plain-English Forensic Translator for Module 4 (Deep Learning & Grad-CAM XAI).
    Currently tailored for Community Forensics ViT-S/16 (`commfor`).
    """

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

    def interpret_detector_frame(
        self, detector_id: str, display_name: str, frame_rec: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not frame_rec.get("evaluated"):
            return {
                "headline": f"Not Evaluated — {display_name} had no valid frame input.",
                "severity": "low",
                "findings": [
                    {
                        "title": "No Input Available",
                        "severity": "info",
                        "detail": "No valid frame or face region from Module 2 could be evaluated.",
                    }
                ],
                "user_guidance": "Inspect a valid frame from Module 2.",
            }

        score = float(frame_rec.get("fake_probability") or 0.0)
        sev = str(frame_rec.get("suspicion_level", "low")).lower()
        pct = self._pct(score)
        level = self._friendly_level(sev)
        faces = frame_rec.get("faces", []) or []
        peak_gid = frame_rec.get("peak_face_id") or "Full Frame"

        if sev == "high":
            headline = (
                f"{level} ({pct}) — {display_name} strongly flagged synthetic AI generation "
                f"patterns on {peak_gid}."
            )
        elif sev == "medium":
            headline = (
                f"{level} ({pct}) — {display_name} detected moderate AI-synthesis cues on {peak_gid}. "
                "Inspect the Grad-CAM heatmap and cross-check with Module 3."
            )
        else:
            headline = (
                f"{level} ({pct}) — {display_name} found the frame patterns consistent with real media."
            )

        findings: List[Dict[str, str]] = []

        # 1. Lập luận riêng cho Community Forensics ViT-S/16 (commfor)
        if detector_id == "commfor":
            if sev in {"high", "medium"}:
                findings.append(
                    {
                        "title": "Full-Frame AI Synthesis Check (Community Forensics ViT-S/16)",
                        "severity": "alert" if sev == "high" else "warning",
                        "detail": (
                            "Evaluates the uncropped whole frame (384x384) to detect generative noise residuals, "
                            "upsampling fingerprints, and global lighting/texture coherence typical of Diffusion or GAN models."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "title": "Full-Frame AI Synthesis Check (Community Forensics ViT-S/16)",
                        "severity": "ok",
                        "detail": (
                            "Evaluated the uncropped 384x384 frame without finding global Diffusion or GAN "
                            "synthesis fingerprints across skin, hair, lighting, or background regions."
                        ),
                    }
                )

        # 2. Lập luận về độ tự tin & biên độ logit
        if faces:
            peak_face = max(faces, key=lambda x: float(x.get("fake_probability", 0.0)))
            p_fake = float(peak_face.get("fake_probability", score))
            p_real = float(peak_face.get("real_probability", 1.0 - p_fake))
            logits = peak_face.get("logits") or {}
            fake_logit = logits.get("fake")

            if p_fake >= 0.85:
                margin_detail = (
                    f"Model confidence is decisive: P(Fake) = {p_fake * 100:.1f}% vs. P(Real) = {p_real * 100:.1f}%"
                    + (f" (raw fake logit: {float(fake_logit):+.2f})." if fake_logit is not None else ".")
                    + " Strong generative indicators outweigh natural camera noise."
                )
                margin_sev = "alert"
            elif p_fake >= 0.65:
                margin_detail = (
                    f"Model leans noticeably toward FAKE (P(Fake) = {p_fake * 100:.1f}% vs. P(Real) = {p_real * 100:.1f}%). "
                    "Verify whether Grad-CAM hotspots match unnatural skin, hair, or background details."
                )
                margin_sev = "warning"
            elif p_fake >= 0.35:
                margin_detail = (
                    f"Borderline confidence (P(Fake) = {p_fake * 100:.1f}% vs. P(Real) = {p_real * 100:.1f}%). "
                    "Heavy social-media compression or low resolution can push scores into this gray zone."
                )
                margin_sev = "warning"
            elif p_fake <= 0.15:
                margin_detail = (
                    f"Model confidence is decisively REAL (P(Real) = {p_real * 100:.1f}% vs. P(Fake) = {p_fake * 100:.1f}%)"
                    + (f" (raw fake logit: {float(fake_logit):+.2f})." if fake_logit is not None else ".")
                )
                margin_sev = "ok"
            else:
                margin_detail = (
                    f"Model leans toward REAL (P(Real) = {p_real * 100:.1f}% vs. P(Fake) = {p_fake * 100:.1f}%), "
                    "with minor low-level anomalies below the manipulation threshold."
                )
                margin_sev = "info"

            findings.append(
                {
                    "title": "Decision Confidence & Logit Margin",
                    "severity": margin_sev,
                    "detail": margin_detail,
                }
            )

            # 3. Lập luận từ vùng chú ý không gian Grad-CAM (xai_metrics)
            xai = peak_face.get("xai_metrics") or {}
            max_act = float(xai.get("max_activation", 0.0))
            mean_act = float(xai.get("mean_activation", 0.0))
            high_area = float(xai.get("high_activation_area_ratio", 0.0))

            if max_act > 0.0:
                if sev in {"high", "medium"}:
                    if high_area >= 0.20:
                        area_note = (
                            f"Suspect activations span broadly ({high_area * 100:.1f}% of the 384x384 view at >60% intensity, "
                            f"mean: {mean_act:.2f}, peak: {max_act:.2f}), pointing to pervasive synthetic textures across the scene."
                        )
                    else:
                        area_note = (
                            f"Grad-CAM pinpointed concentrated high-activation hotspots ({high_area * 100:.1f}% of the view at >60% intensity, "
                            f"peak: {max_act:.2f}, mean: {mean_act:.2f}). Inspect the warm yellow/red regions (e.g., eyes, hairlines, or textured edges)."
                        )
                    xai_sev = "warning"
                else:
                    area_note = (
                        f"Activation remains diffuse relative to the FAKE class "
                        f"(mean: {mean_act:.2f}, >60% area: {high_area * 100:.1f}%), consistent with natural pixel structure."
                    )
                    xai_sev = "ok"

                findings.append(
                    {
                        "title": "Grad-CAM Spatial Attention Evidence",
                        "severity": xai_sev,
                        "detail": area_note,
                    }
                )

        # 4. Cảnh báo cắt viền / góc nghiêng (nếu có)
        unreliable_faces = [
            f.get("global_face_id", "Face")
            for f in faces
            if f.get("unreliable_pose_warning") or float(f.get("padded_ratio", 0.0)) > 0.15
        ]
        if unreliable_faces:
            findings.append(
                {
                    "title": "Edge-Clipped or Extreme Angle Face",
                    "severity": "warning",
                    "detail": (
                        f"Input crop(s) for {', '.join(unreliable_faces)} were partially outside the frame "
                        "or at an extreme angle, which can reduce reliability."
                    ),
                }
            )

        if len(faces) > 1:
            parts = [
                f"{f.get('global_face_id', 'Face')}: {self._pct(f['fake_probability'])}"
                for f in faces
            ]
            findings.append(
                {
                    "title": "Per-Region AI Scores",
                    "severity": "info",
                    "detail": ", ".join(parts),
                }
            )

        guidance = (
            "How to read Community Forensics Grad-CAM: Warm yellow/red zones across the 384x384 full frame "
            "highlight where global AI synthesis traces (skin smoothness, hair edges, or background artifacts) "
            "pushed the model toward FAKE. Compare these hotspots with Module 3's Pixel Grain and 2D-FFT results."
        )

        return {
            "headline": headline,
            "severity": sev,
            "findings": findings,
            "user_guidance": guidance,
        }

    def interpret_manifest(self, m4_manifest: Dict[str, Any]) -> Dict[str, Any]:
        exec_sum = m4_manifest.get("execution_summary", {}) or {}
        detectors_active = m4_manifest.get("detectors_executed", []) or []
        detectors_skipped = m4_manifest.get("detectors_skipped", []) or []
        frames = m4_manifest.get("frames_detail", []) or []
        det_summaries = m4_manifest.get("detector_summaries", {}) or {}

        if not detectors_active:
            if detectors_skipped:
                headline = "No Module 4 AI detectors could run (missing checkpoints or load error)."
                detail = ", ".join(
                    f"{d['detector_id']} ({d['reason']})" for d in detectors_skipped
                )
            else:
                headline = "No Module 4 AI detectors are currently registered/active."
                detail = (
                    "Register a detector in `src/module_4/runner.py` (`DETECTOR_REGISTRY`) "
                    "to enable deep learning inference and Grad-CAM."
                )
            return {
                "headline": headline,
                "severity": "low",
                "findings": [
                    {
                        "title": "Module 4 Status",
                        "severity": "info" if not detectors_skipped else "warning",
                        "detail": detail,
                    }
                ],
                "peak_frames": [],
                "multi_frame": len(frames) > 1,
            }

        overall_sev = str(exec_sum.get("overall_suspicion", "low")).lower()
        primary_score = float(
            exec_sum.get(
                "primary_module4_score",
                exec_sum.get("mean_module4_score", exec_sum.get("max_module4_score", 0.0)),
            )
        )
        max_score = float(exec_sum.get("max_module4_score", 0.0))
        mean_score = float(exec_sum.get("mean_module4_score", 0.0))
        level = self._friendly_level(overall_sev)
        primary_pct = self._pct(primary_score)
        peak_pct = self._pct(max_score)

        if overall_sev == "high":
            headline = (
                f"{level} ({primary_pct}) — Deep learning analysis detected strong AI-generated "
                "indicators in this media."
            )
        elif overall_sev == "medium":
            headline = (
                f"{level} ({primary_pct}) — Deep learning analysis flagged moderate suspicion. "
                "Review the Grad-CAM overlays and Module 3 signal clues together."
            )
        else:
            headline = (
                f"{level} ({primary_pct}) — Active deep learning detector scored this media as likely real."
            )

        findings: List[Dict[str, str]] = []
        eval_frames = [f for f in frames if f.get("evaluated")]
        if len(frames) > 1:
            findings.append(
                {
                    "title": "Frame Coverage",
                    "severity": "info",
                    "detail": (
                        f"Evaluated {len(eval_frames)}/{len(frames)} frames "
                        f"(video average: {self._pct(mean_score)}, highest single frame: {peak_pct})."
                    ),
                }
            )

        if len(detectors_active) == 1:
            only_id = detectors_active[0]
            ds = det_summaries.get(only_id, {})
            findings.append(
                {
                    "title": "Active Forensic Model",
                    "severity": "info",
                    "detail": (
                        f"Evaluated using {ds.get('display_name', only_id.upper())} "
                        f"({ds.get('paradigm', 'Deep Learning Forensics')})."
                    ),
                }
            )

        if detectors_skipped:
            findings.append(
                {
                    "title": "Inactive Detectors",
                    "severity": "info",
                    "detail": "Skipped: "
                    + ", ".join(
                        f"{d['detector_id']} ({d['reason']})" for d in detectors_skipped
                    ),
                }
            )

        peak_frames: List[Dict[str, Any]] = []
        for det_id in detectors_active:
            ds = det_summaries.get(det_id, {})
            det_frames = [
                fr
                for fr in frames
                if fr.get("detectors", {}).get(det_id, {}).get("evaluated")
            ]
            if det_frames:
                best_fr = max(
                    det_frames,
                    key=lambda fr: float(
                        fr["detectors"][det_id]["fake_probability"] or 0.0
                    ),
                )
                peak_frames.append(
                    {
                        "label": ds.get("display_name", det_id.upper()),
                        "frame_index": best_fr["frame_index"],
                        "score": float(best_fr["detectors"][det_id]["fake_probability"]),
                        "severity": str(
                            best_fr["detectors"][det_id]["suspicion_level"]
                        ).lower(),
                        "applicable": True,
                    }
                )

        return {
            "headline": headline,
            "severity": overall_sev,
            "findings": findings,
            "peak_frames": peak_frames,
            "multi_frame": len(frames) > 1,
        }