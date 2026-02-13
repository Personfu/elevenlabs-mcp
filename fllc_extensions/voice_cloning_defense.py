#!/usr/bin/env python3
"""
FLLC Voice Cloning Defense Module
====================================
Counter-measures for AI voice cloning attacks.

Capabilities:
  1. Voice watermark injection — embed imperceptible markers in TTS output
  2. Voiceprint extraction — generate compact speaker identity fingerprint
  3. Clone source identification — detect which TTS engine generated audio
  4. Real-time voice verification — compare live vs known voiceprint

Use cases:
  - Detect social engineering via cloned CEO/exec voices
  - Verify caller identity in sensitive communications
  - Watermark corporate TTS output for attribution
  - Counter vishing (voice phishing) attacks

FLLC 2026 — FU PERSON by PERSON FU
"""

import hashlib
import json
import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# VOICEPRINT — Speaker identity fingerprint
# ═══════════════════════════════════════════════════════════════

@dataclass
class Voiceprint:
    """Compact speaker identity fingerprint."""
    speaker_id: str = ""
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    energy_profile: List[float] = field(default_factory=list)
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0
    segment_rhythm: float = 0.0
    hash_signature: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "pitch_mean": round(self.pitch_mean, 4),
            "pitch_std": round(self.pitch_std, 4),
            "energy_profile": [round(e, 4) for e in self.energy_profile],
            "spectral_centroid": round(self.spectral_centroid, 4),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4),
            "segment_rhythm": round(self.segment_rhythm, 4),
            "hash_signature": self.hash_signature,
        }

    def similarity(self, other: "Voiceprint") -> float:
        """Calculate similarity score between two voiceprints (0.0 - 1.0)."""
        scores = []

        # Pitch comparison
        if self.pitch_mean > 0 and other.pitch_mean > 0:
            pitch_diff = abs(self.pitch_mean - other.pitch_mean)
            scores.append(max(0, 1.0 - pitch_diff / 200.0))

        # Spectral centroid comparison
        if self.spectral_centroid > 0 and other.spectral_centroid > 0:
            sc_diff = abs(self.spectral_centroid - other.spectral_centroid)
            scores.append(max(0, 1.0 - sc_diff / 3000.0))

        # Zero crossing rate comparison
        if self.zero_crossing_rate > 0 and other.zero_crossing_rate > 0:
            zcr_diff = abs(self.zero_crossing_rate - other.zero_crossing_rate)
            scores.append(max(0, 1.0 - zcr_diff / 0.2))

        # Energy profile comparison (cosine similarity)
        if self.energy_profile and other.energy_profile:
            min_len = min(len(self.energy_profile), len(other.energy_profile))
            a = self.energy_profile[:min_len]
            b = other.energy_profile[:min_len]
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x * x for x in a) ** 0.5
            mag_b = sum(x * x for x in b) ** 0.5
            if mag_a > 0 and mag_b > 0:
                scores.append(dot / (mag_a * mag_b))

        return sum(scores) / len(scores) if scores else 0.0


def _read_wav(filepath: str, max_seconds: int = 30) -> Tuple[List[float], int]:
    """Read WAV file, return mono float samples and sample rate."""
    with wave.open(filepath, "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nf = min(wf.getnframes(), sr * max_seconds)
        raw = wf.readframes(nf)

        if sw == 2:
            vals = struct.unpack(f"<{nf * channels}h", raw)
            mx = 32768.0
        elif sw == 1:
            vals = struct.unpack(f"{nf * channels}B", raw)
            mx = 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")

        if channels > 1:
            mono = [sum(vals[i:i + channels]) / channels for i in range(0, len(vals), channels)]
        else:
            mono = list(vals)

        return [s / mx for s in mono], sr


def _zero_crossing_rate(samples: List[float]) -> float:
    """Calculate zero crossing rate."""
    crossings = sum(1 for i in range(1, len(samples))
                    if (samples[i] >= 0) != (samples[i - 1] >= 0))
    return crossings / len(samples) if samples else 0.0


def _spectral_centroid_estimate(samples: List[float], sr: int) -> float:
    """Estimate spectral centroid from time-domain signal."""
    # Simple estimate using zero crossing rate relationship
    zcr = _zero_crossing_rate(samples)
    return zcr * sr / 2.0


def _estimate_pitch(samples: List[float], sr: int) -> Tuple[float, float]:
    """Estimate fundamental frequency using autocorrelation."""
    # Search for pitch in speech range (80-400 Hz)
    min_lag = int(sr / 400)
    max_lag = int(sr / 80)

    if len(samples) < max_lag * 2:
        return 0.0, 0.0

    # Windowed autocorrelation
    window = samples[:max_lag * 2]
    correlations = []
    for lag in range(min_lag, max_lag):
        corr = sum(window[i] * window[i + lag] for i in range(len(window) - lag))
        corr /= (len(window) - lag)
        correlations.append((lag, corr))

    if not correlations:
        return 0.0, 0.0

    # Find peak correlation
    best_lag, best_corr = max(correlations, key=lambda x: x[1])
    pitch = sr / best_lag if best_lag > 0 else 0.0

    # Estimate pitch variation across chunks
    chunk_size = sr  # 1 second chunks
    pitches = []
    for start in range(0, len(samples) - chunk_size, chunk_size // 2):
        chunk = samples[start:start + chunk_size]
        if len(chunk) < max_lag * 2:
            continue
        window = chunk[:max_lag * 2]
        chunk_corrs = []
        for lag in range(min_lag, min(max_lag, len(window) // 2)):
            c = sum(window[i] * window[i + lag] for i in range(len(window) - lag))
            c /= (len(window) - lag)
            chunk_corrs.append((lag, c))
        if chunk_corrs:
            bl, _ = max(chunk_corrs, key=lambda x: x[1])
            if bl > 0:
                pitches.append(sr / bl)

    if pitches:
        mean_p = sum(pitches) / len(pitches)
        std_p = (sum((p - mean_p) ** 2 for p in pitches) / len(pitches)) ** 0.5
        return mean_p, std_p

    return pitch, 0.0


def extract_voiceprint(filepath: str, speaker_id: str = "") -> Voiceprint:
    """Extract a voiceprint fingerprint from a WAV file."""
    samples, sr = _read_wav(filepath)
    vp = Voiceprint()
    vp.speaker_id = speaker_id or hashlib.sha256(filepath.encode()).hexdigest()[:12]

    # Pitch
    vp.pitch_mean, vp.pitch_std = _estimate_pitch(samples, sr)

    # Zero crossing rate
    vp.zero_crossing_rate = _zero_crossing_rate(samples)

    # Spectral centroid estimate
    vp.spectral_centroid = _spectral_centroid_estimate(samples, sr)

    # Energy profile (energy in 10 equal time bins)
    bin_size = len(samples) // 10
    for i in range(10):
        chunk = samples[i * bin_size:(i + 1) * bin_size]
        energy = sum(s * s for s in chunk) / len(chunk) if chunk else 0
        vp.energy_profile.append(energy)

    # Segment rhythm (voiced segment regularity)
    threshold = 0.05
    segments = []
    in_seg = False
    seg_start = 0
    for i, s in enumerate(samples):
        if abs(s) > threshold:
            if not in_seg:
                seg_start = i
                in_seg = True
        else:
            if in_seg:
                segments.append((i - seg_start) / sr)
                in_seg = False

    if len(segments) > 2:
        mean_s = sum(segments) / len(segments)
        std_s = (sum((s - mean_s) ** 2 for s in segments) / len(segments)) ** 0.5
        vp.segment_rhythm = std_s / max(mean_s, 1e-10)

    # Hash signature
    sig_str = f"{vp.pitch_mean:.2f}|{vp.spectral_centroid:.2f}|{vp.zero_crossing_rate:.4f}"
    vp.hash_signature = hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    return vp


# ═══════════════════════════════════════════════════════════════
# CLONE SOURCE IDENTIFICATION
# ═══════════════════════════════════════════════════════════════

TTS_ENGINE_SIGNATURES = {
    "elevenlabs": {
        "pause_cv_range": (0.15, 0.35),
        "zcr_range": (0.03, 0.08),
        "breathing": False,
        "spectral_regularity": "high",
    },
    "resemble_ai": {
        "pause_cv_range": (0.20, 0.40),
        "zcr_range": (0.04, 0.09),
        "breathing": False,
        "spectral_regularity": "medium",
    },
    "xtts_coqui": {
        "pause_cv_range": (0.10, 0.30),
        "zcr_range": (0.05, 0.10),
        "breathing": False,
        "spectral_regularity": "high",
    },
    "bark_suno": {
        "pause_cv_range": (0.25, 0.50),
        "zcr_range": (0.04, 0.12),
        "breathing": True,  # Bark sometimes includes breathing
        "spectral_regularity": "low",
    },
    "tortoise_tts": {
        "pause_cv_range": (0.30, 0.55),
        "zcr_range": (0.03, 0.08),
        "breathing": False,
        "spectral_regularity": "medium",
    },
}


def identify_tts_engine(filepath: str) -> Dict[str, float]:
    """Attempt to identify which TTS engine generated the audio.

    Returns confidence scores for each known engine.
    """
    samples, sr = _read_wav(filepath)
    results = {}

    # Calculate audio features
    zcr = _zero_crossing_rate(samples)

    # Pause analysis
    threshold = 0.01
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
                dur = (i - pause_start) / sr
                if dur > 0.05:
                    pauses.append(dur)
                in_pause = False

    if len(pauses) >= 3:
        mean_p = sum(pauses) / len(pauses)
        std_p = (sum((p - mean_p) ** 2 for p in pauses) / len(pauses)) ** 0.5
        pause_cv = std_p / max(mean_p, 1e-10)
    else:
        pause_cv = 0.5  # Default assumes natural speech

    # Score each engine
    for engine, sig in TTS_ENGINE_SIGNATURES.items():
        score = 0.0
        checks = 0

        # Pause CV match
        lo, hi = sig["pause_cv_range"]
        if lo <= pause_cv <= hi:
            score += 0.4
        elif abs(pause_cv - (lo + hi) / 2) < 0.2:
            score += 0.2
        checks += 1

        # ZCR match
        lo, hi = sig["zcr_range"]
        if lo <= zcr <= hi:
            score += 0.3
        elif abs(zcr - (lo + hi) / 2) < 0.05:
            score += 0.15
        checks += 1

        results[engine] = round(min(1.0, score / 0.7), 3)

    return results


# ═══════════════════════════════════════════════════════════════
# VOICE WATERMARKING (Concept)
# ═══════════════════════════════════════════════════════════════

def inject_watermark(filepath: str, output_path: str, watermark_id: str) -> str:
    """Inject an imperceptible watermark into audio for attribution tracking.

    Uses LSB (Least Significant Bit) modification of audio samples
    to embed a unique identifier without audible artifacts.

    Returns the watermark hash for verification.
    """
    samples, sr = _read_wav(filepath)

    # Generate watermark bit pattern from ID
    wm_hash = hashlib.sha256(watermark_id.encode()).hexdigest()
    wm_bits = "".join(format(int(c, 16), "04b") for c in wm_hash[:32])  # 128 bits

    # Embed pattern by modifying LSB of samples at regular intervals
    embed_interval = max(1, len(samples) // len(wm_bits))

    watermarked = list(samples)
    for i, bit in enumerate(wm_bits):
        idx = i * embed_interval
        if idx >= len(watermarked):
            break

        # Convert to integer domain, modify LSB, convert back
        sample_int = int(watermarked[idx] * 32767)
        if bit == "1":
            sample_int |= 1  # Set LSB
        else:
            sample_int &= ~1  # Clear LSB
        watermarked[idx] = sample_int / 32767.0

    # Write output WAV
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        packed = struct.pack(f"<{len(watermarked)}h",
                             *[int(s * 32767) for s in watermarked])
        wf.writeframes(packed)

    return wm_hash[:32]


def verify_watermark(filepath: str, watermark_id: str) -> float:
    """Verify if a watermark is present in the audio.

    Returns confidence score (0.0 = not found, 1.0 = strong match).
    """
    samples, sr = _read_wav(filepath)

    # Regenerate expected pattern
    wm_hash = hashlib.sha256(watermark_id.encode()).hexdigest()
    wm_bits = "".join(format(int(c, 16), "04b") for c in wm_hash[:32])

    embed_interval = max(1, len(samples) // len(wm_bits))

    # Check LSBs
    matches = 0
    total = 0
    for i, expected_bit in enumerate(wm_bits):
        idx = i * embed_interval
        if idx >= len(samples):
            break
        sample_int = int(samples[idx] * 32767)
        actual_bit = str(sample_int & 1)
        if actual_bit == expected_bit:
            matches += 1
        total += 1

    return matches / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  FLLC Voice Cloning Defense Module v2026")
    print("=" * 55)

    if len(sys.argv) < 3:
        print()
        print("Commands:")
        print("  voiceprint <audio.wav>            Extract voiceprint")
        print("  compare <audio1.wav> <audio2.wav>  Compare voiceprints")
        print("  identify <audio.wav>              Identify TTS engine")
        print("  watermark <in.wav> <out.wav> <id>  Inject watermark")
        print("  verify <audio.wav> <id>            Verify watermark")
        print()
        print("FLLC 2026 -- FU PERSON by PERSON FU")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "voiceprint":
        vp = extract_voiceprint(sys.argv[2])
        print(json.dumps(vp.to_dict(), indent=2))

    elif cmd == "compare":
        vp1 = extract_voiceprint(sys.argv[2], "speaker_1")
        vp2 = extract_voiceprint(sys.argv[3], "speaker_2")
        sim = vp1.similarity(vp2)
        print(f"Similarity: {sim:.4f}")
        if sim > 0.85:
            print("Verdict: SAME SPEAKER (or high-quality clone)")
        elif sim > 0.6:
            print("Verdict: POSSIBLE MATCH — investigate further")
        else:
            print("Verdict: DIFFERENT SPEAKERS")

    elif cmd == "identify":
        results = identify_tts_engine(sys.argv[2])
        print("TTS Engine Confidence Scores:")
        for engine, score in sorted(results.items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 30)
            print(f"  {engine:20s} {score:.3f} [{bar}]")

    elif cmd == "watermark":
        wm = inject_watermark(sys.argv[2], sys.argv[3], sys.argv[4])
        print(f"Watermark injected: {wm}")

    elif cmd == "verify":
        conf = verify_watermark(sys.argv[2], sys.argv[3])
        print(f"Watermark confidence: {conf:.4f}")
        if conf > 0.8:
            print("Watermark VERIFIED")
        elif conf > 0.6:
            print("Watermark PARTIALLY detected (possible degradation)")
        else:
            print("Watermark NOT FOUND")
