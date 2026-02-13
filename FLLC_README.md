```
 ███████╗██╗     ██╗      ██████╗
 ██╔════╝██║     ██║     ██╔════╝
 █████╗  ██║     ██║     ██║
 ██╔══╝  ██║     ██║     ██║
 ██║     ███████╗███████╗╚██████╗
 ╚═╝     ╚══════╝╚══════╝ ╚═════╝
  VOICE INTELLIGENCE — 2026
```

<p align="center">
<img src="https://img.shields.io/badge/FLLC-Voice_Intel-00FFFF?style=for-the-badge&labelColor=0D0D2B"/>
<img src="https://img.shields.io/badge/Deepfake-Detection-FF00FF?style=for-the-badge&labelColor=0D0D2B"/>
<img src="https://img.shields.io/badge/Audio-Forensics-7B2FBE?style=for-the-badge&labelColor=0D0D2B"/>
<img src="https://img.shields.io/badge/TTS-Pipeline-00FFFF?style=for-the-badge&labelColor=0D0D2B"/>
</p>

---

## FLLC Extensions

Custom tools extending the ElevenLabs MCP server for security operations.

| Tool | Description |
|------|-------------|
| `voice_toolkit.py` | Audio analysis, batch TTS, voice clone helper |
| `audio_forensics.py` | Silence detection, spectral analysis, integrity verification, metadata extraction |
| `deepfake_detector.py` | AI voice deepfake detection — spectral discontinuity, pause patterns, breathing analysis, temporal jitter |
| `tts_pipeline.py` | Script-to-podcast batch processor with chapter markers and multi-voice support |

---

## Deepfake Detection (2026 Threat Counter)

Detects AI-generated voices from ElevenLabs, Resemble, XTTS, Bark, and Tortoise TTS:

```bash
python fllc_extensions/deepfake_detector.py suspicious_audio.wav
```

Detection methods:
1. **Spectral discontinuity** — AI stitching artifacts at segment boundaries
2. **Micro-pause analysis** — Unnaturally regular silence patterns
3. **Temporal jitter** — Too-consistent timing in speech segments
4. **Breathing detection** — AI voices lack natural breathing sounds
5. **Formant consistency** — Vocal tract physics violations

---

## Audio Forensics

```bash
python fllc_extensions/audio_forensics.py evidence.wav
```

Outputs: metadata, silence segments, spectral stats, integrity verification.

---

## TTS Pipeline

```bash
python fllc_extensions/tts_pipeline.py script.txt --api-key xi-...
```

Script format:
```
# Chapter 1: Introduction
HOST: Welcome to the FLLC security briefing.
GUEST: Thanks for having me.
---
# Chapter 2: Threat Landscape
HOST: Let's talk about the 2026 threat landscape.
```

---

**FLLC 2026** — FU PERSON by PERSON FU
