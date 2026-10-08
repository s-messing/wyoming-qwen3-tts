import pytest

from wyoming_qwen3_tts.textclean import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**Achtung:** Die Tür ist _offen_.", "Achtung: Die Tür ist offen."),
        ("## Wetter\nEs wird sonnig.", "Wetter Es wird sonnig."),
        ("- Licht an\n- Heizung aus", "Licht an Heizung aus"),
        ("1. Erstens\n2. Zweitens", "Erstens Zweitens"),
        ("Siehe [die Anleitung](https://example.com/x).", "Siehe die Anleitung."),
        ("Mehr unter https://example.com/info dazu.", "Mehr unter dazu."),
        ("Nutze `light.turn_on` dafür.", "Nutze light.turn_on dafür."),
        ("Code:\n```yaml\na: 1\n```\nFertig.", "Code: Fertig."),
        ("Gute Nacht! 🌙😴", "Gute Nacht!"),
        ("~~alt~~ neu", "alt neu"),
        ("| Raum | Temp |\n|---|---|\n| Bad | 21 |", "Raum, Temp Bad, 21"),
        ("> Zitat hier", "Zitat hier"),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected
