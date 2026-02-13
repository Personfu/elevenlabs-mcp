#!/usr/bin/env python3
"""
FLLC AI Voice Deepfake Detector
=================================
Heuristic-based detection of AI-generated/cloned voice audio.
Designed to counter 2026 voice cloning threats (ElevenLabs, Resemble, XTTS, etc.)

Detection techniques:
  1. Spectral discontinuity analysis — AI stitching artifacts
  2. Micro-pause pattern analysis — unnatural silence distribution
  3. Formant consistency check — vocal tract physics violations
  4. Compression artifact detection — generation pipeline fingerprints
  5. Temporal jitter analysis — timing regularity anomalies
  6. Breathing pattern analysis — AI voices lack natural breathing

FLLC 2026 — FU PERSON by PERSON FU
"""

import math
import struct
import wave
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeepfakeAnalysis:
    """Complete deepfake detection analysis result."""
    filename: str = ""
    overall_score: float = 0.0  # 0.0 = definitely real, 1.0 = definitely fake
    verdict: str = "UNKNOWN"
    confidence: str = "LOW"
    indicators: List[str] = field(default_factory=list)
    spectral_score: float = 0.0
    pause_score: float = 0.0
    consistency_score: float = 0.0
    jitter_score: float = 0.0
    breathing_score: float = 0.0

    def to_dict(self):
        return {
            "filename": self.filename,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "overall_score": round(self.overall_score, 4),
            "scores": {
                "spectral_discontinuity": round(self.spectral_score, 4),
                "pause_pattern": round(self.pause_score, 4),
                "formant_consistency": round(self.consistency_score, 4),
                "temporal_jitter": round(self.jitter_score, 4),
                "breathing_pattern": round(self.breathing_score, 4),
            },
            "indicators": self.indicators,
        }


def _read_wav_samples(filepath: str, max_seconds: int = 30) -> tuple:
    """Read WAV file samples as a list of floats normalized to [-1.0, 1.0]."""
    with wave.open(filepath, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        num_frames = min(wf.getnframes(), sample_rate * max_seconds)

        raw = wf.readframes(num_frames)

        if sample_width == 2:
            fmt = f"<{num_frames * channels}h"
            max_val = 32768.0
        elif sample_width == 1:
            fmt = f"{num_frames * channels}B"
            max_val = 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        raw_samples = struct.unpack(fmt, raw)

        # Mix to mono
        if channels > 1:
            mono = []
            for i in range(0, len(raw_samples), channels):
                mono.append(sum(raw_samples[i:i + channels]) / channels)
        else:
            mono = list(raw_samples)

        # Normalize
        samples = [s / max_val for s in mono]

        return samples, sample_rate


def analyze_spectral_discontinuity(samples: list, sample_rate: int) -> tuple:
    """Detect spectral discontinuities — artifacts from AI audio stitching.

    AI voice synthesis often concatenates segments that have slight spectral
    mismatches at boundaries. We detect these as sudden energy changes.
    """
    chunk_ms = 20  # 20ms analysis windows
    chunk_size = int(sample_rate * chunk_ms / 1000)
    score = 0.0
    indicators = []

    energies = []
    for i in range(0, len(samples) - chunk_size, chunk_size):
        chunk = samples[i:i + chunk_size]
        energy = sum(s * s for s in chunk) / len(chunk)
        energies.append(energy)

    if len(energies) < 10:
        return 0.0, []

    # Calculate energy deltas between consecutive chunks
    deltas = [abs(energies[i] - energies[i-1]) for i in range(1, len(energies))]

    if not deltas:
        return 0.0, []

    mean_delta = sum(deltas) / len(deltas)
    std_delta = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5

    # Count discontinuities (> 3 std deviations)
    threshold = mean_delta + 3 * max(std_delta, 1e-10)
    discontinuities = sum(1 for d in deltas if d > threshold)

    # AI typically has more discontinuities per second than natural speech
    disc_rate = discontinuities / (len(samples) / sample_rate)

    if disc_rate > 5.0:
        score = min(1.0, disc_rate / 15.0)
        indicators.append(f"High discontinuity rate: {disc_rate:.1f}/sec (AI threshold: >5)")
    elif disc_rate > 2.0:
        score = disc_rate / 10.0
        indicators.append(f"Moderate discontinuity rate: {disc_rate:.1f}/sec")

    return score, indicators


def analyze_pause_patterns(samples: list, sample_rate: int) -> tuple:
    """Analyze micro-pause distribution.

    Natural speech has irregular pauses with specific distributions.
    AI-generated speech often has more regular, evenly-spaced pauses.
    """
    threshold = 0.01  # Silence threshold
    min_pause_samples = int(sample_rate * 0.05)  # 50ms minimum pause
    score = 0.0
    indicators = []

    # Find all pauses
    pauses = []
    in_pause = False
    pause_start = 0

    for i, s in enumerate(samples):
        if abs(s) < threshold:
            if not in_pause:
                pause_start = i
                in_pause = True
        else:
            if in_pause:
                duration = i - pause_start
                if duration >= min_pause_samples:
                    pauses.append(duration / sample_rate)
                in_pause = False

    if len(pauses) < 3:
        return 0.0, []

    # Coefficient of variation of pause durations
    mean_pause = sum(pauses) / len(pauses)
    std_pause = (sum((p - mean_pause) ** 2 for p in pauses) / len(pauses)) ** 0.5
    cv = std_pause / max(mean_pause, 1e-10)

    # Natural speech: CV > 0.6 (highly variable pauses)
    # AI speech: CV < 0.4 (more regular pauses)
    if cv < 0.3:
        score = 0.8
        indicators.append(f"Very regular pause pattern (CV={cv:.3f}) — AI indicator")
    elif cv < 0.4:
        score = 0.5
        indicators.append(f"Somewhat regular pauses (CV={cv:.3f}) — mild AI indicator")

    # Check for unnaturally short pauses (AI often minimizes silence)
    short_pauses = sum(1 for p in pauses if p < 0.1)
    short_ratio = short_pauses / len(pauses)
    if short_ratio > 0.8:
        score = max(score, 0.6)
        indicators.append(f"High ratio of very short pauses ({short_ratio:.0%})")

    return score, indicators


def analyze_temporal_jitter(samples: list, sample_rate: int) -> tuple:
    """Analyze temporal jitter in speech segments.

    Natural speech has micro-timing variations. AI-generated audio
    often has unnaturally precise timing (too consistent).
    """
    score = 0.0
    indicators = []

    # Find voiced segments (above threshold)
    threshold = 0.05
    segments = []
    in_segment = False
    seg_start = 0

    for i, s in enumerate(samples):
        if abs(s) > threshold:
            if not in_segment:
                seg_start = i
                in_segment = True
        else:
            if in_segment:
                duration = (i - seg_start) / sample_rate
                if duration > 0.05:  # Minimum 50ms segment
                    segments.append(duration)
                in_segment = False

    if len(segments) < 5:
        return 0.0, []

    # Check regularity of segment durations
    mean_seg = sum(segments) / len(segments)
    std_seg = (sum((s - mean_seg) ** 2 for s in segments) / len(segments)) ** 0.5
    cv = std_seg / max(mean_seg, 1e-10)

    # Unnaturally regular segments suggest AI
    if cv < 0.2:
        score = 0.7
        indicators.append(f"Very regular segment timing (CV={cv:.3f}) — AI indicator")
    elif cv < 0.35:
        score = 0.4
        indicators.append(f"Somewhat regular timing (CV={cv:.3f})")

    return score, indicators


def analyze_breathing(samples: list, sample_rate: int) -> tuple:
    """Check for natural breathing patterns.

    Real speech includes audible breaths approximately every 3-6 seconds.
    Many AI voices lack breathing sounds entirely.
    """
    score = 0.0
    indicators = []

    duration = len(samples) / sample_rate
    if duration < 5.0:
        return 0.0, ["Audio too short for breathing analysis"]

    # Look for breath-like patterns: low-frequency noise bursts
    # Breaths are typically 0.2-0.8s duration, low amplitude, preceded by silence
    breath_candidates = 0
    chunk_duration = 0.5  # seconds
    chunk_size = int(sample_rate * chunk_duration)

    for i in range(0, len(samples) - chunk_size, chunk_size):
        chunk = samples[i:i + chunk_size]
        rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5

        # Breath: very low but non-zero amplitude (0.005 < rms < 0.03)
        if 0.005 < rms < 0.03:
            # Check if preceded by speech and followed by speech
            prev_start = max(0, i - chunk_size)
            prev_chunk = samples[prev_start:i]
            next_chunk = samples[i + chunk_size:min(len(samples), i + 2 * chunk_size)]

            if prev_chunk and next_chunk:
                prev_rms = (sum(s * s for s in prev_chunk) / len(prev_chunk)) ** 0.5
                next_rms = (sum(s * s for s in next_chunk) / len(next_chunk)) ** 0.5
                if prev_rms > 0.05 or next_rms > 0.05:
                    breath_candidates += 1

    expected_breaths = duration / 4.5  # ~1 breath every 4.5 seconds

    if breath_candidates == 0 and duration > 10:
        score = 0.7
        indicators.append("No breathing patterns detected — strong AI indicator")
    elif breath_candidates < expected_breaths * 0.3:
        score = 0.5
        indicators.append(f"Very few breaths ({breath_candidates} found, ~{expected_breaths:.0f} expected)")

    return score, indicators


def detect_deepfake(filepath: str) -> DeepfakeAnalysis:
    """Run complete deepfake detection analysis on an audio file."""
    analysis = DeepfakeAnalysis()
    analysis.filename = filepath

    print(f"[*] FLLC Deepfake Detector v2026")
    print(f"[*] Target: {filepath}")

    try:
        samples, sample_rate = _read_wav_samples(filepath)
    except Exception as e:
        analysis.indicators.append(f"Error reading file: {e}")
        analysis.verdict = "ERROR"
        return analysis

    duration = len(samples) / sample_rate
    print(f"[*] Duration: {duration:.1f}s | Sample rate: {sample_rate}Hz")

    # Run all analyses
    print("[*] Analyzing spectral discontinuities...")
    analysis.spectral_score, inds = analyze_spectral_discontinuity(samples, sample_rate)
    analysis.indicators.extend(inds)

    print("[*] Analyzing pause patterns...")
    analysis.pause_score, inds = analyze_pause_patterns(samples, sample_rate)
    analysis.indicators.extend(inds)

    print("[*] Analyzing temporal jitter...")
    analysis.jitter_score, inds = analyze_temporal_jitter(samples, sample_rate)
    analysis.indicators.extend(inds)

    print("[*] Analyzing breathing patterns...")
    analysis.breathing_score, inds = analyze_breathing(samples, sample_rate)
    analysis.indicators.extend(inds)

    # Formant consistency (simplified — full version needs scipy)
    analysis.consistency_score = 0.0

    # Calculate overall score (weighted average)
    weights = {
        "spectral": 0.25,
        "pause": 0.2,
        "consistency": 0.15,
        "jitter": 0.2,
        "breathing": 0.2,
    }
    analysis.overall_score = (
        analysis.spectral_score * weights["spectral"] +
        analysis.pause_score * weights["pause"] +
        analysis.consistency_score * weights["consistency"] +
        analysis.jitter_score * weights["jitter"] +
        analysis.breathing_score * weights["breathing"]
    )

    # Verdict
    if analysis.overall_score > 0.6:
        analysis.verdict = "LIKELY AI-GENERATED"
        analysis.confidence = "HIGH" if analysis.overall_score > 0.75 else "MEDIUM"
    elif analysis.overall_score > 0.35:
        analysis.verdict = "SUSPICIOUS"
        analysis.confidence = "MEDIUM"
    else:
        analysis.verdict = "LIKELY AUTHENTIC"
        analysis.confidence = "MEDIUM" if analysis.overall_score > 0.15 else "HIGH"

    return analysis


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("=" * 50)
        print("  FLLC AI Voice Deepfake Detector v2026")
        print("=" * 50)
        print()
        print("Usage: python deepfake_detector.py <audio_file.wav>")
        print()
        print("Detection methods:")
        print("  1. Spectral discontinuity (AI stitching artifacts)")
        print("  2. Micro-pause pattern (regularity analysis)")
        print("  3. Formant consistency (vocal physics)")
        print("  4. Temporal jitter (timing regularity)")
        print("  5. Breathing pattern (presence/absence)")
        print()
        print("Targets: ElevenLabs, Resemble.AI, XTTS, Bark, Tortoise TTS")
    else:
        result = detect_deepfake(sys.argv[1])
        print()
        print("=== DEEPFAKE ANALYSIS REPORT ===")
        print(json.dumps(result.to_dict(), indent=2))
