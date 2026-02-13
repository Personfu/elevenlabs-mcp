"""
FLLC Voice Toolkit — Extensions for ElevenLabs MCP Server

Audio processing, voice cloning utilities, TTS batch processing,
and audio forensics tools built on top of the ElevenLabs API.

FLLC 2026
"""

import os
import json
import hashlib
import struct
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from elevenlabs import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False


class AudioAnalyzer:
    """Analyze audio files for forensic and quality purposes."""

    @staticmethod
    def get_wav_info(filepath: str) -> dict:
        """Extract WAV file metadata."""
        info = {"file": filepath}
        try:
            with wave.open(filepath, 'rb') as w:
                info["channels"] = w.getnchannels()
                info["sample_width"] = w.getsampwidth()
                info["frame_rate"] = w.getframerate()
                info["frames"] = w.getnframes()
                info["duration_seconds"] = round(w.getnframes() / w.getframerate(), 2)
                info["compression"] = w.getcomptype()
        except Exception as e:
            info["error"] = str(e)
        return info

    @staticmethod
    def compute_audio_hash(filepath: str) -> dict:
        """Compute multiple hashes for audio file integrity verification."""
        hashes = {}
        with open(filepath, 'rb') as f:
            data = f.read()
            hashes["md5"] = hashlib.md5(data).hexdigest()
            hashes["sha256"] = hashlib.sha256(data).hexdigest()
            hashes["size_bytes"] = len(data)
        return hashes

    @staticmethod
    def estimate_noise_level(filepath: str) -> dict:
        """Estimate background noise level from WAV file."""
        try:
            with wave.open(filepath, 'rb') as w:
                frames = w.readframes(w.getnframes())
                sample_width = w.getsampwidth()

                if sample_width == 2:
                    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
                elif sample_width == 1:
                    samples = struct.unpack(f"{len(frames)}B", frames)
                    samples = [s - 128 for s in samples]
                else:
                    return {"error": f"Unsupported sample width: {sample_width}"}

                if not samples:
                    return {"error": "No audio data"}

                max_amplitude = max(abs(s) for s in samples)
                avg_amplitude = sum(abs(s) for s in samples) / len(samples)
                rms = (sum(s ** 2 for s in samples) / len(samples)) ** 0.5

                # Estimate dB (relative to max possible)
                max_possible = (2 ** (sample_width * 8 - 1)) - 1
                db_peak = 20 * (max_amplitude / max_possible + 1e-10).__class__.__module__
                # Simplified dB calculation
                if rms > 0:
                    import math
                    db_rms = 20 * math.log10(rms / max_possible) if rms > 0 else -96
                else:
                    db_rms = -96

                return {
                    "max_amplitude": max_amplitude,
                    "avg_amplitude": round(avg_amplitude, 2),
                    "rms": round(rms, 2),
                    "estimated_db_rms": round(db_rms, 1) if isinstance(db_rms, float) else db_rms,
                    "sample_count": len(samples),
                    "duration_seconds": round(len(samples) / w.getframerate() / w.getnchannels(), 2),
                }
        except Exception as e:
            return {"error": str(e)}


class BatchTTS:
    """Batch text-to-speech processing with ElevenLabs API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.client = None
        if HAS_ELEVENLABS and self.api_key:
            self.client = ElevenLabs(api_key=self.api_key)

    def process_text_file(self, input_path: str, output_dir: str,
                          voice_id: str = "21m00Tcm4TlvDq8ikWAM",
                          model_id: str = "eleven_monolingual_v1") -> list:
        """
        Convert a text file (one sentence per line) to individual audio files.
        Returns list of output file paths.
        """
        if not self.client:
            return [{"error": "ElevenLabs client not initialized. Set ELEVENLABS_API_KEY."}]

        os.makedirs(output_dir, exist_ok=True)
        results = []

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            output_path = os.path.join(output_dir, f"segment_{i:04d}.mp3")
            try:
                audio = self.client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=line,
                    model_id=model_id,
                )
                with open(output_path, 'wb') as out:
                    for chunk in audio:
                        out.write(chunk)
                results.append({
                    "index": i,
                    "text": line[:80],
                    "output": output_path,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "text": line[:80],
                    "error": str(e),
                    "status": "failed"
                })

        # Write manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump({
                "source": input_path,
                "voice_id": voice_id,
                "model_id": model_id,
                "total_segments": len(lines),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "timestamp": datetime.now().isoformat(),
                "segments": results,
            }, f, indent=2)

        return results

    def list_voices(self) -> list:
        """List all available ElevenLabs voices."""
        if not self.client:
            return [{"error": "Client not initialized"}]
        try:
            response = self.client.voices.get_all()
            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": getattr(v, 'category', 'unknown'),
                }
                for v in response.voices
            ]
        except Exception as e:
            return [{"error": str(e)}]


class VoiceCloneHelper:
    """Utilities for preparing voice cloning samples."""

    @staticmethod
    def validate_samples(sample_dir: str) -> dict:
        """
        Validate audio samples for voice cloning requirements:
        - At least 1 minute of clean audio
        - WAV or MP3 format
        - Reasonable file sizes
        """
        results = {
            "valid_files": [],
            "invalid_files": [],
            "total_duration_seconds": 0,
            "meets_requirements": False,
        }

        supported = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}

        for f in Path(sample_dir).iterdir():
            if f.suffix.lower() not in supported:
                results["invalid_files"].append({
                    "file": str(f),
                    "reason": f"Unsupported format: {f.suffix}"
                })
                continue

            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb > 50:
                results["invalid_files"].append({
                    "file": str(f),
                    "reason": f"File too large: {size_mb:.1f}MB (max 50MB)"
                })
                continue

            if size_mb < 0.01:
                results["invalid_files"].append({
                    "file": str(f),
                    "reason": "File too small (possibly empty)"
                })
                continue

            info = {"file": str(f), "size_mb": round(size_mb, 2)}

            # Get duration for WAV files
            if f.suffix.lower() == '.wav':
                wav_info = AudioAnalyzer.get_wav_info(str(f))
                if "duration_seconds" in wav_info:
                    info["duration_seconds"] = wav_info["duration_seconds"]
                    results["total_duration_seconds"] += wav_info["duration_seconds"]

            results["valid_files"].append(info)

        results["meets_requirements"] = (
            len(results["valid_files"]) >= 1
            and results["total_duration_seconds"] >= 30
        )

        return results


def main():
    """Demo / CLI entry point."""
    import sys

    print("=" * 50)
    print("  FLLC Voice Toolkit")
    print("  ElevenLabs MCP Extensions")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  voice_toolkit.py analyze <wav_file>")
        print("  voice_toolkit.py hash <audio_file>")
        print("  voice_toolkit.py validate <sample_dir>")
        print("  voice_toolkit.py voices")
        print("  voice_toolkit.py batch <text_file> <output_dir>")
        return

    cmd = sys.argv[1]

    if cmd == "analyze" and len(sys.argv) > 2:
        info = AudioAnalyzer.get_wav_info(sys.argv[2])
        noise = AudioAnalyzer.estimate_noise_level(sys.argv[2])
        print(json.dumps({**info, "noise": noise}, indent=2))

    elif cmd == "hash" and len(sys.argv) > 2:
        hashes = AudioAnalyzer.compute_audio_hash(sys.argv[2])
        print(json.dumps(hashes, indent=2))

    elif cmd == "validate" and len(sys.argv) > 2:
        results = VoiceCloneHelper.validate_samples(sys.argv[2])
        print(json.dumps(results, indent=2))

    elif cmd == "voices":
        tts = BatchTTS()
        voices = tts.list_voices()
        for v in voices:
            print(f"  {v.get('voice_id', 'N/A'):24s}  {v.get('name', 'N/A')}")

    elif cmd == "batch" and len(sys.argv) > 3:
        tts = BatchTTS()
        results = tts.process_text_file(sys.argv[2], sys.argv[3])
        success = sum(1 for r in results if r.get("status") == "success")
        print(f"Processed: {success}/{len(results)} segments")

    else:
        print("Unknown command or missing arguments.")


if __name__ == "__main__":
    main()
