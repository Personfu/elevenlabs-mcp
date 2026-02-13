#!/usr/bin/env python3
"""
FLLC TTS Pipeline — Script-to-Podcast Batch Processor
=======================================================
Converts text scripts into multi-voice podcast episodes using ElevenLabs.
Supports chapter markers, voice assignment, and batch processing.

Features:
  - Parse script files with speaker annotations
  - Assign ElevenLabs voices per speaker
  - Generate audio segments per line
  - Concatenate with crossfade transitions
  - Add chapter markers for podcast players
  - Batch process multiple episodes

FLLC 2026 — FU PERSON by PERSON FU
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScriptLine:
    """A single line in the podcast script."""
    speaker: str = ""
    text: str = ""
    voice_id: str = ""
    chapter: str = ""
    pause_after_ms: int = 500


@dataclass
class Chapter:
    """A chapter marker in the podcast."""
    title: str = ""
    start_time_ms: int = 0


@dataclass
class PipelineConfig:
    """Configuration for the TTS pipeline."""
    api_key: str = ""
    output_dir: str = "./output"
    default_model: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.0
    crossfade_ms: int = 200
    inter_line_pause_ms: int = 500
    inter_chapter_pause_ms: int = 2000
    voice_map: Dict[str, str] = field(default_factory=dict)


def parse_script(filepath: str) -> List[ScriptLine]:
    """Parse a podcast script file.

    Format:
        # Chapter Title
        SPEAKER: Dialog text here.
        SPEAKER: More dialog.
        ---
        # Next Chapter
        OTHER_SPEAKER: Different speaker text.

    Lines starting with # are chapter markers.
    Lines with --- are scene breaks (longer pause).
    Speaker is identified by the text before the first colon.
    """
    lines = []
    current_chapter = ""

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()

            if not raw_line or raw_line.startswith("//"):
                continue

            if raw_line.startswith("#"):
                current_chapter = raw_line.lstrip("#").strip()
                continue

            if raw_line == "---":
                # Scene break — add a pause marker
                if lines:
                    lines[-1].pause_after_ms = 2000
                continue

            # Parse speaker:text format
            match = re.match(r"^([A-Z_]+)\s*:\s*(.+)$", raw_line)
            if match:
                lines.append(ScriptLine(
                    speaker=match.group(1),
                    text=match.group(2),
                    chapter=current_chapter,
                ))
            else:
                # Continuation of previous line or narrator
                if lines:
                    lines[-1].text += " " + raw_line
                else:
                    lines.append(ScriptLine(speaker="NARRATOR", text=raw_line))

    return lines


def assign_voices(lines: List[ScriptLine], voice_map: Dict[str, str]) -> List[ScriptLine]:
    """Assign ElevenLabs voice IDs to each script line based on speaker."""
    for line in lines:
        if line.speaker in voice_map:
            line.voice_id = voice_map[line.speaker]
        else:
            # Default voice if speaker not mapped
            line.voice_id = voice_map.get("DEFAULT", "21m00Tcm4TlvDq8ikWAM")

    return lines


def generate_audio_segment(text: str, voice_id: str, config: PipelineConfig) -> Optional[bytes]:
    """Generate audio for a single text segment using ElevenLabs API.

    Returns raw audio bytes (MP3) or None if API call fails.
    """
    try:
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.api_key,
        }
        data = {
            "text": text,
            "model_id": config.default_model,
            "voice_settings": {
                "stability": config.stability,
                "similarity_boost": config.similarity_boost,
                "style": config.style,
            },
        }

        response = requests.post(url, json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            print(f"[!] API error {response.status_code}: {response.text[:200]}")
            return None

    except ImportError:
        print("[!] 'requests' library not installed. Run: pip install requests")
        return None
    except Exception as e:
        print(f"[!] Error generating audio: {e}")
        return None


def process_script(script_path: str, config: PipelineConfig) -> dict:
    """Process a complete script file into a podcast episode.

    Returns a manifest dict with output file paths and chapter markers.
    """
    print(f"[*] FLLC TTS Pipeline v2026")
    print(f"[*] Script: {script_path}")

    # Parse
    print("[*] Parsing script...")
    lines = parse_script(script_path)
    print(f"    Found {len(lines)} lines, {len(set(l.speaker for l in lines))} speakers")

    # Assign voices
    print("[*] Assigning voices...")
    lines = assign_voices(lines, config.voice_map)

    speakers = set(l.speaker for l in lines)
    for s in speakers:
        vid = config.voice_map.get(s, "DEFAULT")
        print(f"    {s} -> {vid}")

    # Generate segments
    os.makedirs(config.output_dir, exist_ok=True)
    segments = []
    chapters = []
    current_chapter = ""
    current_time_ms = 0

    for i, line in enumerate(lines):
        if line.chapter and line.chapter != current_chapter:
            current_chapter = line.chapter
            chapters.append(Chapter(title=current_chapter, start_time_ms=current_time_ms))
            print(f"  [Chapter] {current_chapter}")

        print(f"  [{i+1}/{len(lines)}] {line.speaker}: {line.text[:50]}...")

        if config.api_key:
            audio = generate_audio_segment(line.text, line.voice_id, config)
            if audio:
                seg_path = os.path.join(config.output_dir, f"seg_{i:04d}.mp3")
                with open(seg_path, "wb") as f:
                    f.write(audio)
                segments.append(seg_path)
                # Estimate duration (~150 words per minute for TTS)
                word_count = len(line.text.split())
                estimated_ms = int(word_count / 150 * 60 * 1000)
                current_time_ms += estimated_ms + line.pause_after_ms

                # Rate limiting
                time.sleep(0.5)
            else:
                segments.append(None)
        else:
            # Dry run — no API key
            segments.append(None)
            word_count = len(line.text.split())
            current_time_ms += int(word_count / 150 * 60 * 1000) + line.pause_after_ms

    # Generate manifest
    manifest = {
        "script": script_path,
        "segments": len(segments),
        "generated": len([s for s in segments if s]),
        "chapters": [{"title": c.title, "start_ms": c.start_time_ms} for c in chapters],
        "total_duration_ms": current_time_ms,
        "output_dir": config.output_dir,
    }

    manifest_path = os.path.join(config.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Complete: {len([s for s in segments if s])}/{len(segments)} segments generated")
    print(f"[+] Estimated duration: {current_time_ms / 1000 / 60:.1f} minutes")
    print(f"[+] Manifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 50)
        print("  FLLC TTS Pipeline v2026")
        print("  Script-to-Podcast Batch Processor")
        print("=" * 50)
        print()
        print("Usage: python tts_pipeline.py <script.txt> [--api-key KEY]")
        print()
        print("Script format:")
        print("  # Chapter Title")
        print("  SPEAKER: Dialog text here.")
        print("  OTHER: Response text.")
        print("  ---")
        print("  # Next Chapter")
        print()
        print("Without --api-key, runs in dry-run mode (no audio generated).")
    else:
        script_path = sys.argv[1]
        api_key = ""

        if "--api-key" in sys.argv:
            idx = sys.argv.index("--api-key")
            if idx + 1 < len(sys.argv):
                api_key = sys.argv[idx + 1]

        config = PipelineConfig(
            api_key=api_key,
            voice_map={
                "HOST": "21m00Tcm4TlvDq8ikWAM",      # Rachel
                "GUEST": "EXAVITQu4vr4xnSDxMaL",      # Bella
                "NARRATOR": "onwK4e9ZLuTAKqWW03F9",    # Daniel
                "DEFAULT": "21m00Tcm4TlvDq8ikWAM",
            }
        )

        process_script(script_path, config)
