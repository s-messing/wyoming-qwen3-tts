import numpy as np

# Qwen3-TTS 12Hz tokenizer output format
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1

# ISO 639-1 code -> language name expected by Qwen3-TTS
QWEN_LANGUAGES: dict[str, str] = {
    "de": "German",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(QWEN_LANGUAGES)


def to_pcm(audio: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (samples * 32767).astype(np.int16).tobytes()


def language_code(language: str | None, default: str) -> str:
    """Normalize 'de', 'de-DE', 'de_DE' to a supported ISO code, else the default."""
    if language:
        code = language.lower().replace("_", "-").split("-")[0]
        if code in SUPPORTED_LANGUAGES:
            return code
    return default
