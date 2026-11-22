from __future__ import annotations

from typing import Any, Dict, List


class Module4ForensicTranslator:
    """
    Plain-English Forensic Translator for Module 4 (Deep Learning & Grad-CAM XAI).
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
                "headline": f"Not Evaluated — No face was detected in this frame for {display_name}.",
                "severity": "low",
                "findings": [
                    {
                        "title": "No Face Available",
                        "severity": "info",
                        "detail": "Face-based AI detectors require at least one aligned face from Module 2.",
                    }
                ],
                "user_guidance": "Inspect a frame where a face is clearly detected.",
            }

        score = float(frame_rec.get("fake_probability") or 0.0)
        sev = str(frame_rec.get("suspicion_level", "low")).lower()
        pct = self._pct(score)
        level = self._friendly_level(sev)
        faces = frame_rec.get("faces", []) or []
        peak_gid = frame_rec.get("peak_face_id") or "Detected Face"

        if sev == "high":
            headline = (
                f"{level} ({pct}) — {display_name} strongly flagged synthetic or manipulated "
                f"patterns on {peak_gid}."
            )
        elif sev == "medium":
            headline = (
                f"{level} ({pct}) — {display_name} detected moderate manipulation cues on {peak_gid}. "
                "Inspect the Grad-CAM heatmap and cross-check with Module 3."
            )
        else:
            headline = (
                f"{level} ({pct}) — {display_name} found the face frequency patterns consistent with real media."
            )

        findings: List[Dict[str, str]] = []
        if len(faces) > 1:
            parts = [
                f"{f.get('global_face_id', 'Face')}: {self._pct(f['fake_probability'])}"
                for f in faces
            ]
            findings.append(
                {
                    "title": "Per-Face AI Scores",
                    "severity": "info",
                    "detail": ", ".join(parts),
                }
            )

        if detector_id == "f3net" and faces:
            peak_face = max(faces, key=lambda x: float(x["fake_probability"]))
            bands = peak_face.get("fad_band_energy", {}) or {}
            high_b = float(bands.get("high_band_share", 0.0))
            mid_b = float(bands.get("mid_band_share", 0.0))
            low_b = float(bands.get("low_band_share", 0.0))
            findings.append(
                {
                    "title": "DCT Frequency-Band Energy Share (FAD)",
                    "severity": "info",
                    "detail": (
                        f"Low-freq: {low_b * 100:.1f}%, Mid-freq: {mid_b * 100:.1f}%, "
                        f"High-freq: {high_b * 100:.1f}%. F3Net inspects all bands simultaneously "
                        "to catch frequency traces missed in RGB pixels."
                    ),
                }
            )

        guidance = (
            "How to read the Grad-CAM image: Warm yellow/red zones show which facial regions "
            "(eyes, mouth, nose, or jawline) pushed the AI model most strongly toward predicting FAKE. "
            "Compare these hotspots with Module 3's Face Border and Pixel Grain heatmaps."
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

        if not detectors_active:
            return {
                "headline": "No Module 4 AI detectors could run (missing checkpoints or disabled).",
                "severity": "low",
                "findings": [
                    {
                        "title": "Skipped Detectors",
                        "severity": "warning",
                        "detail": ", ".join(
                            f"{d['detector_id']} ({d['reason']})" for d in detectors_skipped
                        )
                        or "None configured.",
                    }
                ],
                "peak_frames": [],
                "multi_frame": len(frames) > 1,
            }

        overall_sev = str(exec_sum.get("overall_suspicion", "low")).lower()
        max_score = float(exec_sum.get("max_module4_score", 0.0))
        mean_score = float(exec_sum.get("mean_module4_score", 0.0))
        level = self._friendly_level(overall_sev)
        peak_pct = self._pct(max_score)

        if overall_sev == "high":
            headline = (
                f"{level} ({peak_pct}) — Deep learning analysis detected strong AI manipulation "
                "indicators in this media."
            )
        elif overall_sev == "medium":
            headline = (
                f"{level} ({peak_pct}) — Deep learning models flagged moderate suspicion. "
                "Review the Grad-CAM overlays and Module 3 signal clues together."
            )
        else:
            headline = (
                f"{level} ({peak_pct}) — Active deep learning detector(s) scored this media as likely real."
            )

        findings: List[Dict[str, str]] = []
        eval_frames = [f for f in frames if f.get("evaluated")]
        if len(frames) > 1:
            findings.append(
                {
                    "title": "Frame Coverage",
                    "severity": "info",
                    "detail": (
                        f"Evaluated {len(eval_frames)}/{len(frames)} frames with faces "
                        f"(average peak score: {self._pct(mean_score)}, highest frame score: {peak_pct})."
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
        det_summaries = m4_manifest.get("detector_summaries", {}) or {}
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
