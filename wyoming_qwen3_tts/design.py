"""Create voices from a text description (Qwen3-TTS VoiceDesign).

VoiceDesign re-invents the voice on every request, so it is not used for serving. Instead a
designed clip becomes the fixed reference that the Base model clones (see engine.py).
"""

import logging
import re
import shutil
import wave
from pathlib import Path

import numpy as np

from .audio import CHANNELS, QWEN_LANGUAGES, SAMPLE_RATE, SAMPLE_WIDTH, to_pcm

_LOGGER = logging.getLogger(__name__)

CANDIDATES_DIR = "_candidates"
REFERENCE_TEXT = {
    "de": "Hallo, ich bin dein Assistent für zu Hause. Ich kümmere mich um Licht, Heizung und alles, was sonst so anfällt.",
    "en": "Hello, I am your home assistant. I take care of the lights, the heating and everything else that comes up.",
}


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(to_pcm(audio))


def design(
    voices_path: Path,
    *,
    name: str,
    description: str,
    model_id: str,
    language: str,
    candidates: int = 3,
    text: str | None = None,
) -> list[Path]:
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    text = text or REFERENCE_TEXT.get(language, REFERENCE_TEXT["en"])
    out_dir = voices_path / CANDIDATES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Loading %s", model_id)
    model = FasterQwen3TTS.from_pretrained(model_id)

    paths = []
    for i in range(1, candidates + 1):
        torch.manual_seed(i)  # different but reproducible candidates
        chunks = [
            np.asarray(chunk, dtype=np.float32).reshape(-1)
            for chunk, _sr, _timing in model.generate_voice_design_streaming(
                text=text, instruct=description, language=QWEN_LANGUAGES[language], chunk_size=12
            )
        ]
        wav = out_dir / f"{name}_{i}.wav"
        write_wav(wav, np.concatenate(chunks))
        wav.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
        paths.append(wav)
        _LOGGER.info("Candidate %d: %s", i, wav)

    (out_dir / f"{name}.description").write_text(description + "\n", encoding="utf-8")
    return paths


def pick(voices_path: Path, name: str, candidate: int) -> Path:
    """Promote a candidate to voices/<name>.wav + .txt and remove the other candidates."""
    out_dir = voices_path / CANDIDATES_DIR
    wav = out_dir / f"{name}_{candidate}.wav"
    if not wav.exists():
        raise FileNotFoundError(f"Candidate not found: {wav}")

    target = voices_path / f"{name}.wav"
    shutil.move(wav, target)
    shutil.move(wav.with_suffix(".txt"), target.with_suffix(".txt"))
    description = out_dir / f"{name}.description"
    if description.exists():
        shutil.move(description, voices_path / f"{name}.description")

    candidate_re = re.compile(rf"{re.escape(name)}_\d+\.(?:wav|txt)")
    stale = [p for p in out_dir.iterdir() if candidate_re.fullmatch(p.name)]
    for path in [*stale, *voices_path.glob(f"{name}.*.pt")]:
        path.unlink()
    if not any(out_dir.iterdir()):
        out_dir.rmdir()
    return target
