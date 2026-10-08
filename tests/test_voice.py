from pathlib import Path

import pytest

from wyoming_qwen3_tts.audio import language_code
from wyoming_qwen3_tts.voice import resolve_voice, scan_voices


def make_voice(path: Path, name: str, text: str | None = "Hallo.") -> None:
    (path / f"{name}.wav").write_bytes(b"RIFF")
    if text is not None:
        (path / f"{name}.txt").write_text(text, encoding="utf-8")


def test_scan_skips_voice_without_transcript(tmp_path: Path) -> None:
    make_voice(tmp_path, "max", "Hallo, ich bin Max.\n")
    make_voice(tmp_path, "orphan", None)
    voices = scan_voices(tmp_path)
    assert [v.name for v in voices] == ["max"]
    assert voices[0].ref_text == "Hallo, ich bin Max."


def test_resolve_voice(tmp_path: Path) -> None:
    make_voice(tmp_path, "anna")
    make_voice(tmp_path, "max")
    assert resolve_voice(tmp_path, None).name == "anna"
    assert resolve_voice(tmp_path, "max").name == "max"
    assert resolve_voice(tmp_path, "max.wav").name == "max"
    with pytest.raises(ValueError, match="not found"):
        resolve_voice(tmp_path, "nobody")


def test_resolve_voice_without_voices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No voices"):
        resolve_voice(tmp_path, None)


@pytest.mark.parametrize(
    ("language", "expected"),
    [(None, "de"), ("", "de"), ("de", "de"), ("de-DE", "de"), ("en_US", "en"), ("FR", "fr"), ("pl", "de")],
)
def test_language_code(language: str | None, expected: str) -> None:
    assert language_code(language, "de") == expected
