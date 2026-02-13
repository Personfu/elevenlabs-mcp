# FLLC Extensions — ElevenLabs MCP Server

Custom audio processing and voice toolkit extensions built on the official ElevenLabs MCP server.

## FLLC Extensions

| Tool | Description |
|------|-------------|
| `AudioAnalyzer` | WAV metadata, noise estimation, file integrity hashing |
| `BatchTTS` | Bulk text-to-speech from text files with manifest generation |
| `VoiceCloneHelper` | Sample validation for voice cloning requirements |

## Quick Start

```bash
# Set your ElevenLabs API key
export ELEVENLABS_API_KEY="your_key_here"

# Analyze an audio file
python fllc_extensions/voice_toolkit.py analyze sample.wav

# Compute integrity hashes
python fllc_extensions/voice_toolkit.py hash recording.mp3

# Validate voice cloning samples
python fllc_extensions/voice_toolkit.py validate ./samples/

# List available voices
python fllc_extensions/voice_toolkit.py voices

# Batch convert text to speech
python fllc_extensions/voice_toolkit.py batch script.txt ./output/
```

## Original MCP Server

This repo is forked from [elevenlabs/elevenlabs-mcp](https://github.com/elevenlabs/elevenlabs-mcp). See the original `README.md` for MCP server setup and configuration.

---

*FLLC 2026*
