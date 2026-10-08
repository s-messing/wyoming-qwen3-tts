# wyoming-qwen3-tts

A Wyoming protocol server for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech, built for Home Assistant and tuned for German.

Based on [lmoe/wyoming-xtts](https://github.com/lmoe/wyoming-xtts). The Wyoming handling, streaming and Zeroconf code come from there; the XTTS engine was replaced by Qwen3-TTS.

## Why

Piper's German voices sound robotic, and XTTS v2 is old and has a non-commercial licence. Qwen3-TTS (Apache-2.0, January 2026) speaks good German and, through [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), streams its first audio in about 0.2 s on a consumer GPU.

## Features

- Native Wyoming protocol with Zeroconf discovery, following wyoming-piper.
- Bidirectional streaming: text streams in from the LLM, audio streams out sentence by sentence.
- **Voices from a description.** You describe a voice in plain words, pick the best of a few candidates, and the server clones that fixed clip on every request so the voice stays consistent.
- **German text normalization.** Numbers, decimals, dates, times, units, currency and abbreviations are spelled out before synthesis ("21,5 °C" → "einundzwanzig Komma fünf Grad"). Qwen3-TTS garbles or derails on raw symbols and digits.
- **Markdown and emoji are stripped.** Home Assistant passes raw LLM output to TTS.
- **German-aware sentence splitting.** "z. B." and "am 3. Oktober" stay in one piece.
- **Fair scheduling for several rooms.** Requests take turns sentence by sentence on the GPU, so a second room doesn't wait for the first room's whole answer.
- Generation is capped by text length, so a runaway generation stops early.
- A fixed seed makes the same sentence always sound the same.

## Quick Start

```bash
mkdir -p /path/to/qwen3_data

docker run -d \
  --gpus all \
  -p 10200:10200 \
  --name wyoming-qwen3-tts \
  -v /path/to/qwen3_data:/data \
  ghcr.io/s-messing/wyoming-qwen3-tts
```

The models (about 4.5 GB for 1.7B-Base) are downloaded to `/data/hf` on first start.

Then add the server to Home Assistant:

1. Settings → Devices & services → Add integration → Wyoming Protocol → enter IP and port (default 10200), or use the auto-detected `wyoming-qwen3-tts` entry.
2. Settings → Voice assistants → your assistant → Text-to-speech → wyoming-qwen3-tts.
3. Pick your voice.

## Creating a voice

A voice is a reference clip `voices/<name>.wav` plus its exact transcript `voices/<name>.txt`. The easiest way to create one is from a description:

```bash
# 1. Generate three candidates speaking a German reference sentence
docker run --rm --gpus all -v /path/to/qwen3_data:/data ghcr.io/s-messing/wyoming-qwen3-tts \
  design --name max \
  --description "A calm male voice in his forties, native German speaker with standard High German pronunciation, clear, friendly and slightly deep."

# 2. Listen to /path/to/qwen3_data/voices/_candidates/max_{1,2,3}.wav, then keep the best one
docker run --rm -v /path/to/qwen3_data:/data ghcr.io/s-messing/wyoming-qwen3-tts pick --name max --candidate 2

# 3. Restart the server to load the new voice
docker restart wyoming-qwen3-tts
```

- `design` loads the 1.7B VoiceDesign model (about 4.3 GB VRAM) only for this command. Run it while the server is stopped if VRAM is tight.
- Descriptions can be written in English or German.
- `--text` changes the reference sentence. `--candidates` changes how many candidates are generated.

You can also use a recording of a real voice. Put `name.wav` (3–10 s of clean speech) and `name.txt` (exactly what is said) into `voices/`.

On first start the server turns each voice into a clone prompt and caches it as `voices/<name>.<model>.pt`. Delete that file to rebuild it.

## Assets

Mount a folder or volume to `/data`:

```
/data/
├── voices/   # <name>.wav + <name>.txt per voice, plus cached <name>.<model>.pt prompts
└── hf/       # Hugging Face model cache
```

## Configuration

Environment variables only, no config files.

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_URI` | `tcp://0.0.0.0:10200` | Server address |
| `QWEN3_ASSETS` | (local) `./assets`, (docker) `/data` | Assets directory |
| `QWEN3_ZEROCONF` | `wyoming-qwen3-tts` | Zeroconf service name (set empty to disable) |
| `QWEN3_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model. `Qwen/Qwen3-TTS-12Hz-0.6B-Base` saves about 1 GB VRAM at slightly lower quality |
| `QWEN3_DESIGN_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Model used by `design` |
| `QWEN3_OFFLINE` | `false` | Use only the local model cache, never download |
| `QWEN3_LANGUAGE` | `de` | Language when Home Assistant does not send one: de, en, fr, es, it, pt, ru, zh, ja, ko |
| `QWEN3_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

### Synthesis parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_TEMPERATURE` | `0.5` | Sampling temperature. Lower is more uniform and slightly more "assistant-like"; the model default 0.9 is more expressive |
| `QWEN3_TOP_K` | `50` | Top-k sampling |
| `QWEN3_TOP_P` | `1.0` | Nucleus sampling threshold |
| `QWEN3_REPETITION_PENALTY` | `1.05` | Repetition penalty |
| `QWEN3_CHUNK_SIZE` | `4` | Codec steps per streamed audio chunk (12.5 steps = 1 s). Lower means faster first audio |
| `QWEN3_MIN_SEGMENT_CHARS` | `20` | Minimum characters before a segment is synthesized |
| `QWEN3_SEED` | `42` | The same text always produces the same audio. Set `QWEN3_SEED=""` for random variation |

CLI arguments work too (`--model`, `--temperature 0.7`, …).

## Language

Home Assistant currently does not send the selected language to Wyoming TTS servers. This server therefore uses `QWEN3_LANGUAGE` (default German) unless a request carries a language. German text normalization only runs for German.

If you want your own replacement rules on top, for example for names or Denglisch, [TTS Proxy](https://github.com/Thyraz/tts-proxy) can sit in Home Assistant between the assistant and this server.

## Requirements

- An NVIDIA GPU with bf16 support. The image is built for CUDA 12.8 (Volta through Blackwell / RTX 50xx).
- Docker with nvidia-container-toolkit, or podman with CDI.
- VRAM: about 4.3 GB for 1.7B-Base and about 3.3 GB for 0.6B-Base, measured on an RTX 5060 Ti.
- About 5 GB of disk space per model.

Measured on an RTX 5060 Ti 16 GB (1.7B, German): about 0.2 s to first audio, and about 2.2× faster than real time.

## License

MIT. Qwen3-TTS models are Apache-2.0. `de_tables.py` adapts tables from [Godelaune/Kokoro-82M-ONNX-German-Martin](https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin) (Apache-2.0).
