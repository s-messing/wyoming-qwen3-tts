import re
from collections.abc import Generator

from sentence_stream import SentenceBoundaryDetector

# sentence-stream splits on ". " + capital letter. German capitalizes nouns, so ordinals
# ("am 3. Oktober", "der 2. Platz") and abbreviations ("z. B.", "bzw. Strom") get split.
# A detected sentence ending like this is held and joined with the next one.
_HOLD_ABBREVIATIONS = (
    "z",
    "b",
    "d",
    "h",
    "u",
    "a",
    "ca",
    "bzw",
    "vgl",
    "ggf",
    "evtl",
    "inkl",
    "zzgl",
    "ggü",
    "bspw",
    "Nr",
    "Abb",
    "Abs",
    "Kap",
    "S",
    "Str",
    "Dr",
    "Prof",
    "St",
    "Tsd",
    "Mio",
    "Mrd",
    "Std",
    "Min",
    "Sek",
)
_HOLD_RE = re.compile(r"(?:\b\d{1,2}|(?<![\w.])(?:" + "|".join(_HOLD_ABBREVIATIONS) + r"))\.$", re.IGNORECASE)


def _should_hold(sentence: str) -> bool:
    return _HOLD_RE.search(sentence.rstrip()) is not None


class BufferedSegmenter:
    def __init__(self, min_chars: int = 20) -> None:
        self._sbd = SentenceBoundaryDetector()
        self._min_chars = min_chars
        self._buffer = ""

    def add_chunk(self, text: str) -> Generator[str, None, None]:
        for sentence in self._sbd.add_chunk(text):
            self._buffer += (" " if self._buffer else "") + sentence.strip()
            if len(self._buffer) >= self._min_chars and not _should_hold(self._buffer):
                yield self._buffer
                self._buffer = ""

    def finish(self) -> str:
        remaining = self._sbd.finish()
        if remaining and remaining.strip():
            self._buffer += (" " if self._buffer else "") + remaining.strip()
        result = self._buffer
        self._buffer = ""
        return result
