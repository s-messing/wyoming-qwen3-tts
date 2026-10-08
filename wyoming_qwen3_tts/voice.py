import logging
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    """A voice is a reference clip plus its exact transcript (needed for ICL cloning)."""

    name: str
    wav: Path
    txt: Path

    @property
    def ref_text(self) -> str:
        return self.txt.read_text(encoding="utf-8").strip()


def scan_voices(voices_path: Path) -> list[Voice]:
    voices = []
    for wav in sorted(voices_path.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            _LOGGER.warning("Skipping voice '%s': transcript %s is missing", wav.stem, txt.name)
            continue
        voices.append(Voice(name=wav.stem, wav=wav, txt=txt))
    return voices


def resolve_voice(voices_path: Path, voice_name: str | None) -> Voice:
    voices = scan_voices(voices_path)
    if not voices:
        raise ValueError(f"No voices found in {voices_path} (each voice needs <name>.wav and <name>.txt)")
    if voice_name is None:
        return voices[0]
    for voice in voices:
        if voice_name in (voice.name, voice.wav.name):
            return voice
    raise ValueError(f"Voice '{voice_name}' not found in {voices_path}")
