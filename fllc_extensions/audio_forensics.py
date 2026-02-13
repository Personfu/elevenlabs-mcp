#!/usr/bin/env python3
"""
FLLC Audio Forensics Toolkit
==============================
Audio analysis tools for security investigations and forensic work.

Features:
  - Silence detection and segmentation
  - Spectral analysis (frequency distribution)
  - Audio integrity verification (hash, metadata)
  - Metadata extraction and stripping
  - Audio fingerprinting
  - Steganography detection hints

FLLC 2026 — FU PERSON by PERSON FU
"""

import hashlib
import json
import os
import struct
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AudioMetadata:
    """Extracted audio file metadata."""
    filename: str = ""
    format: str = ""
    channels: int = 0
    sample_rate: int = 0
    bit_depth: int = 0
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    sha256: str = ""
    created: str = ""
    modified: str = ""


@dataclass
class SilenceSegment:
    """A detected silence segment in audio."""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class ForensicReport:
    """Complete forensic analysis report."""
    metadata: Optional[AudioMetadata] = None
    silence_segments: list = field(default_factory=list)
    spectral_stats: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    anomalies: list = field(default_factory=list)
    analysis_timestamp: str = ""


def calculate_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Calculate hash of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_metadata(filepath: str) -> AudioMetadata:
    """Extract metadata from a WAV audio file."""
    meta = AudioMetadata()
    meta.filename = os.path.basename(filepath)
    meta.file_size_bytes = os.path.getsize(filepath)
    meta.sha256 = calculate_file_hash(filepath)

    stat = os.stat(filepath)
    meta.modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

    if filepath.lower().endswith(".wav"):
        meta.format = "WAV"
        try:
            with wave.open(filepath, "rb") as wf:
                meta.channels = wf.getnchannels()
                meta.sample_rate = wf.getframerate()
                meta.bit_depth = wf.getsampwidth() * 8
                frames = wf.getnframes()
                meta.duration_seconds = frames / float(meta.sample_rate)
        except Exception as e:
            meta.format = f"WAV (error: {e})"
    else:
        meta.format = os.path.splitext(filepath)[1].upper().lstrip(".")

    return meta


def detect_silence(filepath: str, threshold: float = 0.01, min_duration: float = 0.5) -> list:
    """Detect silence segments in a WAV file.

    Args:
        filepath: Path to WAV file.
        threshold: Amplitude threshold below which audio is considered silence (0.0-1.0).
        min_duration: Minimum duration in seconds for a segment to be considered silence.

    Returns:
        List of SilenceSegment objects.
    """
    segments = []

    try:
        with wave.open(filepath, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()

            # Read all frames
            raw = wf.readframes(num_frames)

            # Determine format
            if sample_width == 1:
                fmt = f"{num_frames * channels}B"
                max_val = 255.0
            elif sample_width == 2:
                fmt = f"<{num_frames * channels}h"
                max_val = 32768.0
            else:
                return segments  # Unsupported bit depth

            samples = struct.unpack(fmt, raw)

            # Analyze in chunks
            chunk_size = int(sample_rate * 0.05)  # 50ms chunks
            in_silence = False
            silence_start = 0.0

            for i in range(0, len(samples), chunk_size * channels):
                chunk = samples[i:i + chunk_size * channels]
                if not chunk:
                    break

                # Calculate RMS amplitude
                rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5 / max_val
                current_time = i / (sample_rate * channels)

                if rms < threshold:
                    if not in_silence:
                        silence_start = current_time
                        in_silence = True
                else:
                    if in_silence:
                        duration = current_time - silence_start
                        if duration >= min_duration:
                            segments.append(SilenceSegment(
                                start_seconds=round(silence_start, 3),
                                end_seconds=round(current_time, 3),
                                duration_seconds=round(duration, 3)
                            ))
                        in_silence = False

            # Handle trailing silence
            if in_silence:
                end_time = num_frames / sample_rate
                duration = end_time - silence_start
                if duration >= min_duration:
                    segments.append(SilenceSegment(
                        start_seconds=round(silence_start, 3),
                        end_seconds=round(end_time, 3),
                        duration_seconds=round(duration, 3)
                    ))

    except Exception as e:
        print(f"[!] Error analyzing {filepath}: {e}")

    return segments


def spectral_analysis(filepath: str) -> dict:
    """Basic spectral analysis of a WAV file.

    Returns frequency distribution statistics without requiring numpy/scipy.
    Uses zero-crossing rate as a proxy for dominant frequency.
    """
    stats = {"zero_crossing_rate": 0.0, "rms_amplitude": 0.0, "peak_amplitude": 0.0,
             "dynamic_range_db": 0.0, "dc_offset": 0.0}

    try:
        with wave.open(filepath, "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = min(wf.getnframes(), sample_rate * 10)  # Analyze first 10s
            channels = wf.getnchannels()

            raw = wf.readframes(num_frames)

            if sample_width == 2:
                fmt = f"<{num_frames * channels}h"
            elif sample_width == 1:
                fmt = f"{num_frames * channels}B"
            else:
                return stats

            samples = list(struct.unpack(fmt, raw))

            # Mix to mono if stereo
            if channels > 1:
                mono = []
                for i in range(0, len(samples), channels):
                    mono.append(sum(samples[i:i + channels]) // channels)
                samples = mono

            if not samples:
                return stats

            # Zero crossing rate
            crossings = sum(
                1 for i in range(1, len(samples))
                if (samples[i] >= 0) != (samples[i-1] >= 0)
            )
            stats["zero_crossing_rate"] = round(crossings / len(samples) * sample_rate / 2, 1)

            # RMS amplitude
            max_val = 32768.0 if sample_width == 2 else 255.0
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / max_val
            stats["rms_amplitude"] = round(rms, 6)

            # Peak amplitude
            peak = max(abs(s) for s in samples) / max_val
            stats["peak_amplitude"] = round(peak, 6)

            # Dynamic range
            import math
            if rms > 0 and peak > 0:
                stats["dynamic_range_db"] = round(20 * math.log10(peak / max(rms, 1e-10)), 2)

            # DC offset
            dc = sum(samples) / len(samples) / max_val
            stats["dc_offset"] = round(dc, 6)

    except Exception as e:
        stats["error"] = str(e)

    return stats


def verify_integrity(filepath: str, expected_hash: str = None) -> dict:
    """Verify audio file integrity.

    Checks:
      - File hash matches expected (if provided)
      - WAV header consistency
      - File size matches header-declared data
    """
    result = {
        "file_exists": os.path.exists(filepath),
        "hash_sha256": "",
        "hash_match": None,
        "header_valid": False,
        "size_consistent": False,
        "anomalies": []
    }

    if not result["file_exists"]:
        result["anomalies"].append("File does not exist")
        return result

    result["hash_sha256"] = calculate_file_hash(filepath)

    if expected_hash:
        result["hash_match"] = result["hash_sha256"].lower() == expected_hash.lower()
        if not result["hash_match"]:
            result["anomalies"].append("Hash mismatch — file may be tampered")

    if filepath.lower().endswith(".wav"):
        try:
            with wave.open(filepath, "rb") as wf:
                result["header_valid"] = True
                declared_frames = wf.getnframes()
                expected_size = declared_frames * wf.getnchannels() * wf.getsampwidth() + 44  # WAV header
                actual_size = os.path.getsize(filepath)

                if abs(actual_size - expected_size) < 100:  # Allow small header variance
                    result["size_consistent"] = True
                else:
                    result["size_consistent"] = False
                    result["anomalies"].append(
                        f"Size mismatch: expected ~{expected_size}B, got {actual_size}B"
                    )
        except Exception as e:
            result["header_valid"] = False
            result["anomalies"].append(f"Invalid WAV header: {e}")

    return result


def full_forensic_analysis(filepath: str) -> ForensicReport:
    """Run complete forensic analysis on an audio file."""
    report = ForensicReport()
    report.analysis_timestamp = datetime.now().isoformat()

    print(f"[*] FLLC Audio Forensics v2026")
    print(f"[*] Target: {filepath}")

    print("[*] Extracting metadata...")
    report.metadata = extract_metadata(filepath)

    print("[*] Detecting silence segments...")
    report.silence_segments = detect_silence(filepath)

    print("[*] Running spectral analysis...")
    report.spectral_stats = spectral_analysis(filepath)

    print("[*] Verifying integrity...")
    report.integrity = verify_integrity(filepath)
    report.anomalies = report.integrity.get("anomalies", [])

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 50)
        print("  FLLC Audio Forensics Toolkit v2026")
        print("=" * 50)
        print()
        print("Usage: python audio_forensics.py <audio_file>")
        print("  Supported: WAV files")
        print()
        print("Functions:")
        print("  - Metadata extraction (hash, channels, duration)")
        print("  - Silence detection and segmentation")
        print("  - Spectral analysis (ZCR, RMS, dynamic range)")
        print("  - File integrity verification")
    else:
        report = full_forensic_analysis(sys.argv[1])
        print()
        print("=== REPORT ===")
        print(json.dumps({
            "metadata": report.metadata.__dict__ if report.metadata else {},
            "silence_segments": [s.__dict__ for s in report.silence_segments],
            "spectral_stats": report.spectral_stats,
            "integrity": report.integrity,
            "anomalies": report.anomalies,
        }, indent=2))
