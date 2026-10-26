"""
src/normalizer.py - Unicode, Devanagari, and whitespace text normalizer.

Ensures text that represents the same semantic content compares identically:
1. Unicode NFKC normalization
2. Devanagari Nukta canonicalization (precomposed vs decomposed)
3. Zero-Width characters and format marks (ZWJ, ZWNJ, ZWSP, BOM, LTR/RTL)
4. Devanagari punctuation & danda normalization
5. Whitespace and non-printable control character standardization
"""

import unicodedata
import re

# Mapping decomposed Devanagari consonants + Nukta (U+093C) to precomposed Unicode forms
NUKTA_DECOMPOSED_TO_PRECOMPOSED = {
    "\u0915\u093C": "\u0958",  # क़ (Ka + Nukta -> Qa)
    "\u0916\u093C": "\u0959",  # ख़ (Kha + Nukta -> Khha)
    "\u0917\u093C": "\u095A",  # ग़ (Ga + Nukta -> Ghha)
    "\u091C\u093C": "\u095B",  # ज़ (Ja + Nukta -> Za)
    "\u0921\u093C": "\u095C",  # ड़ (Dda + Nukta -> Ddda)
    "\u0922\u093C": "\u095D",  # ढ़ (Ddha + Nukta -> Rha)
    "\u092B\u093C": "\u095E",  # फ़ (Pha + Nukta -> Fa)
    "\u092F\u093C": "\u095F",  # य़ (Ya + Nukta -> Yya)
}

# Precomposed to decomposed mapping for canonical comparison
NUKTA_PRECOMPOSED_TO_CANONICAL = {v: k for k, v in NUKTA_DECOMPOSED_TO_PRECOMPOSED.items()}

# Invisible characters to remove unconditionally
INVISIBLE_CHARS_REGEX = re.compile(
    r"[\u200B\u200E\u200F\u202A-\u202E\u2060\uFEFF]"  # ZWSP, LTR/RTL marks, embeddings, Word Joiner, BOM
)

# Control characters except newline and tab
CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Multiple spaces or unusual whitespace
WHITESPACE_REGEX = re.compile(r"[^\S\n\r]+")

# Repeated newlines (more than 2 consecutive)
MULTI_NEWLINES_REGEX = re.compile(r"\n{3,}")

# Pipe symbol surrounded by Devanagari or space normalized to Danda
PIPE_TO_DANDA_REGEX = re.compile(r"(?<=[\u0900-\u097F\s])\|(?=[\u0900-\u097F\s]|$)")

# Double pipe to double danda
DOUBLE_PIPE_REGEX = re.compile(r"\|\|")


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFKC normalization."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_devanagari_nukta(text: str) -> str:
    """Canonicalize Devanagari nukta representations to precomposed forms."""
    for decomposed, precomposed in NUKTA_DECOMPOSED_TO_PRECOMPOSED.items():
        text = text.replace(decomposed, precomposed)
    # Deduplicate repeated nuktas if any
    text = re.sub(r"\u093C+", "\u093C", text)
    return text


def clean_invisible_chars(text: str) -> str:
    """Remove invisible formatting and directionality marks."""
    # Remove BOM, ZWSP, and directional marks
    text = INVISIBLE_CHARS_REGEX.sub("", text)
    
    # Clean up orphaned or redundant ZWJ / ZWNJ (keep only if adjacent to virama / halant)
    # ZWJ: \u200D, ZWNJ: \u200C, Halant: \u094D
    text = re.sub(r"(?<!\u094D)[\u200C\u200D]", "", text)
    text = re.sub(r"[\u200C\u200D](?![\u0900-\u097F])", "", text)
    
    # Remove non-printable control characters
    text = CONTROL_CHARS_REGEX.sub("", text)
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize quotes, dashes, and Devanagari danda."""
    # Quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("`", "'").replace("´", "'")
    
    # Dashes
    text = text.replace("—", "-").replace("–", "-")
    
    # Double pipe to double danda
    text = DOUBLE_PIPE_REGEX.sub("॥", text)
    # Pipe surrounded by Devanagari to single danda
    text = PIPE_TO_DANDA_REGEX.sub("।", text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """Standardize line breaks, collapse runs of spaces, and trim."""
    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking and unusual spaces with standard ASCII space
    text = text.replace("\u00A0", " ").replace("\u202F", " ")
    # Collapse consecutive horizontal whitespace
    text = WHITESPACE_REGEX.sub(" ", text)
    # Collapse 3+ newlines to 2
    text = MULTI_NEWLINES_REGEX.sub("\n\n", text)
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def normalize_text(text: str) -> str:
    """
    Full text normalization pipeline:
    1. Unicode NFKC
    2. Invisible & control characters removal
    3. Devanagari nukta canonicalization
    4. Punctuation & danda normalization
    5. Whitespace standardization
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = clean_invisible_chars(text)
    text = normalize_devanagari_nukta(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


def normalize_for_comparison(text: str) -> str:
    """
    Aggressive normalization used strictly for exact / near-deduplication comparison:
    Lowercases Latin, strips punctuation and extra whitespace.
    """
    normalized = normalize_text(text)
    normalized = normalized.lower()
    # Strip punctuation for comparison
    normalized = re.sub(r"[^\w\s\u0900-\u097F]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
