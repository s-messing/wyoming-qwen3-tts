"""German text normalization for TTS: numbers, dates, times, units and abbreviations to words.

Qwen3-TTS garbles or derails on raw "21,5 °C" or "03.11.2026" but reads written-out text reliably.
"""

import logging
import re

from num2words import num2words

from .de_tables import (
    ABBREVIATIONS,
    FEMININE_SUFFIXES,
    MONTHS,
    ORDINAL_E_CONTEXT,
    ORDINAL_N_CONTEXT,
    UNITS,
)

_LOGGER = logging.getLogger(__name__)

_ABBREVIATIONS = tuple((re.compile(p), r) for p, r in ABBREVIATIONS)
_UNIT_BY_SYMBOL = {re.sub(r"\\(.)", r"\1", p): (s, pl, g) for p, s, pl, g in UNITS}
_MONTH_RE = "|".join(MONTHS)
# A number: optional sign, digits with optional thousands dots, optional decimal comma
_NUM = r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?|-?\d+(?:,\d+)?"
_UNIT_RE = "|".join(p for p, *_ in UNITS)
_DASH = "[-\u2013]"  # hyphen or en dash
_ONE = {"m": "ein", "n": "ein", "f": "eine"}
# "im Jahr 1990" / "seit 1985": read 1100-1999 as years ("neunzehnhundertneunzig")
_YEAR_CONTEXT = frozenset({"jahr", "jahre", "jahres", "seit", "bis", "von", "anno", "ab", "vor", "nach", "um", "im"})


def _int_words(n: int) -> str:
    return str(num2words(n, lang="de"))


def _number_words(raw: str) -> str:
    """'21,5' -> 'einundzwanzig Komma fünf', '1.250' -> 'eintausendzweihundertfünfzig', '-3' -> 'minus drei'."""
    raw = raw.strip()
    sign = ""
    if raw.startswith(("-", "\u2212")):  # hyphen or minus sign
        sign, raw = "minus ", raw[1:]
    integer, _, decimals = raw.replace(".", "").partition(",")
    words = _int_words(int(integer))
    if decimals:
        words += " Komma " + " ".join(_int_words(int(d)) for d in decimals)
    return sign + words


def _ordinal(n: int, before: str) -> str:
    base = str(num2words(n, lang="de", to="ordinal"))  # "dritte"
    word = before.lower()
    if word in ORDINAL_N_CONTEXT:
        return base + "n"
    if word in ORDINAL_E_CONTEXT:
        return base
    return base + "r"


def _word_before(text: str, pos: int) -> str:
    m = re.search(r"(\w+)\W*$", text[:pos])
    return m.group(1) if m else ""


def _year_words(y: str) -> str:
    year = int(y)
    if len(y) == 2:
        year += 2000
    return str(num2words(year, lang="de", to="year"))


def _sub_dates(text: str) -> str:
    # ISO 2026-11-03
    def iso(m: re.Match[str]) -> str:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m[0]
        return f"{_ordinal(d, _word_before(m.string, m.start()))} {MONTHS[mo - 1]} {_year_words(str(y))}"

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", iso, text)

    # 03.11.2026 / 3.11. / 03.11.26
    def numeric(m: re.Match[str]) -> str:
        d, mo = int(m[1]), int(m[2])
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m[0]
        out = f"{_ordinal(d, _word_before(m.string, m.start()))} {MONTHS[mo - 1]}"
        return out + (f" {_year_words(m[3])}" if m[3] else "")

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(?:(\d{4}|\d{2})\b)?", numeric, text)

    # 5.-7. Mai
    def day_range(m: re.Match[str]) -> str:
        before = _word_before(m.string, m.start())
        return f"{_ordinal(int(m[1]), before)} bis {_ordinal(int(m[2]), 'den')} {m[3]}"

    text = re.sub(rf"\b(\d{{1,2}})\.\s*{_DASH}\s*(\d{{1,2}})\.\s*({_MONTH_RE})\b", day_range, text)

    # 3. Oktober
    def day_month(m: re.Match[str]) -> str:
        return f"{_ordinal(int(m[1]), _word_before(m.string, m.start()))} {m[2]}"

    return re.sub(rf"\b(\d{{1,2}})\.\s*({_MONTH_RE})\b", day_month, text)


def _clock(h: int, mi: int) -> str:
    hour = "ein" if h == 1 else _int_words(h)
    return f"{hour} Uhr" + (f" {_int_words(mi)}" if mi else "")


def _sub_times(text: str) -> str:
    def with_uhr(m: re.Match[str]) -> str:
        return _clock(int(m[1]), int(m[2] or 0))

    text = re.sub(r"\b([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\s*Uhr\b", with_uhr, text)

    def bare(m: re.Match[str]) -> str:
        return _clock(int(m[1]), int(m[2]))

    return re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?\b", bare, text)


def _sub_ordinals(text: str) -> str:
    # "der 2. Platz", "im 3. Stock": only with a preceding article/preposition, so a
    # sentence-final number ("Es sind 7. Danach ...") is left alone.
    def repl(m: re.Match[str]) -> str:
        before = _word_before(m.string, m.start())
        if before.lower() in ORDINAL_N_CONTEXT | ORDINAL_E_CONTEXT:
            return _ordinal(int(m[1]), before) + " "
        if re.search(r"(?:^|[,;:(])\s*$", m.string[: m.start()]):  # list item or start: "3. Stock"
            return _ordinal(int(m[1]), "") + " "
        return m[0]

    return re.sub(r"\b(\d{1,3})\.\s+(?=[A-ZÄÖÜ][a-zäöüß])", repl, text)


def _sub_money(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        euros = int(m[1].replace(".", ""))
        cents = int(m[2]) if m[2] else 0
        if euros == 0 and cents:
            return f"{_int_words(cents)} Cent"
        return f"{_int_words(euros)} Euro" + (f" {_int_words(cents)}" if cents else "")

    text = re.sub(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?\s*(?:€|EUR\b|Euro\b)", repl, text)
    return re.sub(r"€\s*(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?", repl, text)


def _sub_units(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        num, unit = m[1], m[2]
        singular, plural, gender = _UNIT_BY_SYMBOL[unit]
        if num in ("1", "-1"):
            return f"{'minus ' if num[0] == '-' else ''}{_ONE[gender]} {singular}"
        return f"{_number_words(num)} {plural}"

    return re.sub(rf"({_NUM})\s*({_UNIT_RE})(?![\w/²³])", repl, text)


def _sub_ranges(text: str) -> str:
    return re.sub(rf"(?<![\w,.])({_NUM})\s*{_DASH}\s*(?=\d)", r"\1 bis ", text)


def _one_before_noun(m: re.Match[str]) -> str:
    noun = m[1]
    return ("eine " if noun.lower().endswith(FEMININE_SUFFIXES) else "ein ") + noun


def _sub_numbers(text: str) -> str:
    text = re.sub(r"(?<![\w,.])1\s+([A-ZÄÖÜ][a-zäöüß]+)", _one_before_noun, text)
    text = re.sub(r"(?<![\w,.])-(?=\d)", "minus ", text)

    def year(m: re.Match[str]) -> str:
        if _word_before(m.string, m.start()).lower() not in _YEAR_CONTEXT:
            return m[0]
        return _year_words(m[0])

    text = re.sub(r"(?<![\w,.])1[1-9]\d{2}(?![\w,]|\.\d)", year, text)

    def number(m: re.Match[str]) -> str:
        return _number_words(m[0])

    return re.sub(r"(?<![\w,])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?(?!\w)", number, text)


def _sub_abbreviations(text: str) -> str:
    for pattern, repl in _ABBREVIATIONS:
        text = pattern.sub(repl, text)
    return text


def _sub_symbols(text: str) -> str:
    text = re.sub(r"\s+&\s+", " und ", text)
    text = re.sub(r"\s+\+\s+", " plus ", text)
    text = re.sub(r"\s+=\s+", " gleich ", text)
    return text.replace("°", " Grad")


def normalize_de(raw: str) -> str:
    try:
        text = _sub_abbreviations(raw)
        text = _sub_dates(text)
        text = _sub_times(text)
        text = _sub_ordinals(text)
        text = _sub_money(text)
        text = _sub_ranges(text)
        text = _sub_units(text)
        text = _sub_numbers(text)
        text = _sub_symbols(text)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        _LOGGER.exception("German normalization failed, using raw text: %r", raw)
        return raw


def normalize(text: str, language: str) -> str:
    if language == "de":
        return normalize_de(text)
    return text
