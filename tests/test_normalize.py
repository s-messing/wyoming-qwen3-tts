import pytest

from wyoming_qwen3_tts.de_normalize import normalize, normalize_de


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Sentences that garbled or derailed Qwen3-TTS in the listening tests
        (
            "Im Schlafzimmer sind es gerade 21,5 °C bei 48 % Luftfeuchtigkeit.",
            "Im Schlafzimmer sind es gerade einundzwanzig Komma fünf Grad bei achtundvierzig Prozent Luftfeuchtigkeit.",
        ),
        (
            "Mit Höchstwerten um 14 Grad und leichtem Regen ab 16:30 Uhr.",
            "Mit Höchstwerten um vierzehn Grad und leichtem Regen ab sechzehn Uhr dreißig.",
        ),
        ("Alles klar, ich stelle einen Timer auf 7 Minuten.", "Alles klar, ich stelle einen Timer auf sieben Minuten."),
        (
            "Der Termin ist am 03.11.2026 um 9:15 Uhr, ca. 2,3 km von hier entfernt.",
            "Der Termin ist am dritten November zweitausendsechsundzwanzig um neun Uhr fünfzehn, "
            "circa zwei Komma drei Kilometer von hier entfernt.",
        ),
        # Money
        ("Das kostet 1.250 €.", "Das kostet eintausendzweihundertfünfzig Euro."),
        ("Nur 5,99 € pro Monat.", "Nur fünf Euro neunundneunzig pro Monat."),
        ("Das sind 0,50 €.", "Das sind fünfzig Cent."),
        # Ordinals and dates
        ("Der 2. Platz geht an Anna.", "Der zweite Platz geht an Anna."),
        ("Wohnung 4, 3. Stock.", "Wohnung vier, dritter Stock."),
        ("Er wohnt im 3. Stock.", "Er wohnt im dritten Stock."),
        ("Vom 5.\u20137. Mai ist Urlaub.", "Vom fünften bis siebten Mai ist Urlaub."),  # en dash
        ("Vom 5.-7. Mai ist Urlaub.", "Vom fünften bis siebten Mai ist Urlaub."),
        ("Heute ist der 3. Oktober.", "Heute ist der dritte Oktober."),
        ("Seit 2024-10-08 läuft alles.", "Seit achten Oktober zweitausendvierundzwanzig läuft alles."),
        ("Im Jahr 1990 war das anders.", "Im Jahr neunzehnhundertneunzig war das anders."),
        ("Es sind 7. Danach kommt nichts.", "Es sind sieben. Danach kommt nichts."),
        # Times
        ("Es ist 7:05.", "Es ist sieben Uhr fünf."),
        ("Um 1 Uhr nachts.", "Um ein Uhr nachts."),
        ("Ab 20 Uhr.", "Ab zwanzig Uhr."),
        # Units, signs, ranges
        ("Draußen sind -3 °C.", "Draußen sind minus drei Grad."),
        (
            "Die PV liefert 3,2 kW, heute 12,4 kWh bei 230 V.",
            "Die PV liefert drei Komma zwei Kilowatt, heute zwölf Komma vier Kilowattstunden bei zweihundertdreißig Volt.",
        ),
        ("Gestern war es 1 kWh.", "Gestern war es eine Kilowattstunde."),
        ("Wind mit 25 km/h.", "Wind mit fünfundzwanzig Kilometer pro Stunde."),
        ("Es regnet 5-7 mm.", "Es regnet fünf bis sieben Millimeter."),
        ("Luftdruck 1013 hPa.", "Luftdruck eintausenddreizehn Hektopascal."),
        # Abbreviations and counting
        ("Das ist z. B. anders, d. h. besser.", "Das ist zum Beispiel anders, das heißt besser."),
        ("Wohnung Nr. 5 bzw. 6.", "Wohnung Nummer fünf beziehungsweise sechs."),
        ("1 Lampe und 1 Fernseher sind an.", "eine Lampe und ein Fernseher sind an."),
        # Untouched
        ("Okay, ich habe das Licht eingeschaltet.", "Okay, ich habe das Licht eingeschaltet."),
    ],
)
def test_normalize_de(raw: str, expected: str) -> None:
    assert normalize_de(raw) == expected


def test_only_german_is_normalized() -> None:
    assert normalize("It is 21.5 °C.", "en") == "It is 21.5 °C."


def test_never_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_text: str) -> str:
        raise ValueError("boom")

    monkeypatch.setattr("wyoming_qwen3_tts.de_normalize._sub_dates", boom)
    assert normalize_de("am 03.11.2026") == "am 03.11.2026"
