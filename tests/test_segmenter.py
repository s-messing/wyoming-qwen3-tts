import pytest

from wyoming_qwen3_tts.segmenter import BufferedSegmenter


def segment(text: str, chunk: int = 7, min_chars: int = 1) -> list[str]:
    """Feed text in small chunks, like an LLM streaming tokens."""
    seg = BufferedSegmenter(min_chars=min_chars)
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out.extend(seg.add_chunk(text[i : i + chunk]))
    rest = seg.finish()
    if rest:
        out.append(rest)
    return out


@pytest.mark.parametrize(
    "sentence",
    [
        "Das ist z. B. am Wochenende so.",
        "Wir sehen uns am 3. Oktober wieder.",
        "Er belegt den 2. Platz im Rennen.",
        "Es kostet ca. Zwanzig Euro.",
        "Wir sparen bzw. Strom und Wasser.",
        "Das ist teuer, d. h. Ärger droht.",
        "Termin am 03.11.2026 um 9:15 Uhr, ca. 2,3 km entfernt.",
        "Das liegt in der Str. Nummer fünf.",
    ],
)
def test_german_abbreviations_and_ordinals_are_not_split(sentence: str) -> None:
    assert segment(sentence + " Danach kommt der nächste Satz.") == [sentence, "Danach kommt der nächste Satz."]


def test_sentence_ends_still_split() -> None:
    text = "Das Licht ist an. Die Heizung läuft! Soll ich noch etwas tun? Es ist 16 Uhr. Bitte."
    assert segment(text) == ["Das Licht ist an.", "Die Heizung läuft!", "Soll ich noch etwas tun?", "Es ist 16 Uhr.", "Bitte."]


def test_min_chars_merges_short_sentences() -> None:
    assert segment("Ja. Okay. Das Licht ist jetzt an.", min_chars=20) == ["Ja. Okay. Das Licht ist jetzt an."]


def test_held_sentence_is_flushed_on_finish() -> None:
    assert segment("Wir sehen uns am 3.") == ["Wir sehen uns am 3."]
