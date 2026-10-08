"""Strip markdown and emoji from LLM output. Home Assistant passes raw LLM text to TTS."""

import re

import emoji

_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"```.*?(?:```|$)", re.DOTALL), " "),  # code blocks
    (re.compile(r"`([^`]*)`"), r"\1"),  # inline code
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links -> text
    (re.compile(r"https?://\S+"), ""),  # bare URLs
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),  # headings
    (re.compile(r"^\s{0,3}>\s?", re.MULTILINE), ""),  # blockquotes
    (re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE), ""),  # list markers
    (re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*$", re.MULTILINE), ""),  # table separator rows
    (re.compile(r"^[ \t]*\|[ \t]*|[ \t]*\|[ \t]*$", re.MULTILINE), ""),  # table row edges
    (re.compile(r"[ \t]*\|[ \t]*"), ", "),  # table cells
    (re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE), ""),  # horizontal rules
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),  # bold
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),  # italic *x*
    (re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"), r"\1"),  # italic _x_
    (re.compile(r"~~(.+?)~~"), r"\1"),  # strikethrough
    (re.compile(r"[*#]+"), ""),  # leftover markers
)


def clean_text(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"^[ \t,]+|[ \t,]+$", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()
