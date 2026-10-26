"""
src/filter.py - Quality and noise filtering for Indic LLM training corpora.

Filters out:
- Garbled encoding (mojibake, replacement characters)
- Markup, javascript, and website boilerplate
- Character and word repetition spam
- Truncated / empty documents

Deliberately preserves legitimate unusual text:
- Short proverbs and idioms
- Verse and poetry (high line-break to word ratio)
- Data tables / sports scorecards / numeric statistics
"""

import re
from typing import Tuple, Optional
from collections import Counter

# Mojibake patterns (UTF-8 bytes misdecoded as Windows-1252 / ISO-8859-1)
MOJIBAKE_REGEX = re.compile(r"(?:Ã¢â‚¬|à¤¹à¤¿|à¤|Ã©|Ã¨|Ã¼|Ã±|â€œ|â€\x9d|â€˜|â€™)")

# HTML / XML tags
HTML_TAG_REGEX = re.compile(r"<[^>]+>")

# URLs
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")

# Repeated characters (5 or more identical characters consecutively)
CHAR_REPETITION_REGEX = re.compile(r"(.)\1{4,}")

# Common boilerplate phrases
BOILERPLATE_PATTERNS = [
    re.compile(r"\bterms (?:and|&) conditions\b", re.I),
    re.compile(r"\ball rights reserved\b", re.I),
    re.compile(r"\bcookie policy\b", re.I),
    re.compile(r"\bprivacy policy\b", re.I),
    re.compile(r"\bunsubscribe from (?:this|our) newsletter\b", re.I),
    re.compile(r"\bjavascript is disabled\b", re.I),
    re.compile(r"\benable javascript to run this app\b", re.I),
]

# Devanagari script regex
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
WORD_REGEX = re.compile(r"\b[\w\u0900-\u097F]+\b")


def check_garbled_encoding(text: str) -> Tuple[bool, Optional[str]]:
    """Check for replacement characters (\ufffd) or UTF-8 mojibake patterns."""
    # Unicode replacement character
    replacement_count = text.count("\ufffd")
    if replacement_count > 0:
        if replacement_count > 2 or (replacement_count / max(len(text), 1)) > 0.01:
            return True, "garbled_mojibake_replacement_chars"
            
    # Common double-encoded UTF-8 artifacts
    mojibake_matches = MOJIBAKE_REGEX.findall(text)
    if len(mojibake_matches) >= 2:
        return True, "garbled_utf8_misencoding"
        
    return False, None


def check_markup_and_code(text: str) -> Tuple[bool, Optional[str]]:
    """Check for excessive markup, HTML boilerplate, or code blocks."""
    html_tags = HTML_TAG_REGEX.findall(text)
    tag_char_len = sum(len(t) for t in html_tags)
    
    # If HTML tags make up more than 15% of the document
    if tag_char_len > 0 and (tag_char_len / len(text)) > 0.15:
        return True, "markup_excessive_html"
        
    # Check boilerplate phrases
    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(text):
            # If boilerplate is a major part of a short text
            if len(text.split()) < 30:
                return True, "boilerplate_website_notice"
                
    return False, None


def check_repetition_and_spam(text: str) -> Tuple[bool, Optional[str]]:
    """Check for character-level spam or degenerate n-gram loops."""
    # 5+ repeated characters (e.g. 'aaaaaa', '??????')
    if CHAR_REPETITION_REGEX.search(text):
        return True, "spam_char_repetition"
        
    # Link spam
    urls = URL_REGEX.findall(text)
    if len(urls) >= 3 and (len(urls) / max(len(text.split()), 1)) > 0.10:
        return True, "spam_excessive_urls"
        
    words = WORD_REGEX.findall(text.lower())
    if len(words) >= 15:
        # Check 3-gram diversity
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            # Repeated copypasta or generated loops have extremely low unique ratio
            if unique_ratio < 0.22:
                return True, "spam_ngram_repetition_loop"
                
    return False, None


def is_legitimate_unusual_text(text: str) -> bool:
    """
    Identifies legitimate but unusual texts that should NOT be dropped:
    - Short proverbs (short, high meaningful word ratio, grammatical)
    - Poetry / verse (rhyme, high newline ratio, balanced vocabulary)
    - Tables / scorecards (valid entity nouns + numeric structure)
    """
    words = WORD_REGEX.findall(text)
    word_count = len(words)
    
    # 1. Short proverbs (e.g. "नाच न जाने आँगन टेढ़ा।", "सत्यमेव जयते।")
    if 2 <= word_count <= 10 and len(text) >= 10:
        # If text ends with valid punctuation or danda and contains no junk
        has_punctuation = any(p in text for p in ["।", ".", "!", "?", "॥"])
        devanagari_chars = len(DEVANAGARI_REGEX.findall(text))
        if has_punctuation and (devanagari_chars >= 5 or any(w.isalpha() for w in words)):
            return True
            
    # 2. Verse / Poetry: High newline frequency, multiple short lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 3 and word_count >= 8:
        avg_line_len = sum(len(l.split()) for l in lines) / len(lines)
        # Typical poetry: 3 to 12 words per line with unique vocabulary
        if 2 <= avg_line_len <= 14:
            unique_words_ratio = len(set(w.lower() for w in words)) / word_count
            if unique_words_ratio >= 0.50:
                return True
                
    # 3. Numeric tables / sports scorecards (e.g. Rohit 47 (31), Kohli 76 (59))
    digit_count = sum(c.isdigit() for c in text)
    if digit_count >= 5 and word_count >= 4:
        digit_ratio = digit_count / max(len(text), 1)
        # Legitimate scorecards/tables have up to 45% digits, but still have entity words
        if digit_ratio <= 0.45 and len(set(words)) >= 3:
            return True
            
    return False


def filter_quality(text: str) -> Tuple[bool, Optional[str]]:
    """
    Assess text quality.
    Returns:
        (is_passed, rejection_reason)
        is_passed: True if document passes quality filter, False if dropped.
        rejection_reason: string description if dropped, None if kept.
    """
    stripped = text.strip()
    
    # Empty or near-empty
    if len(stripped) == 0:
        return False, "empty_document"
        
    # Garbled encoding / Mojibake check first
    is_garbled, reason = check_garbled_encoding(stripped)
    if is_garbled:
        return False, reason
        
    # Markup & Boilerplate check
    is_markup, reason = check_markup_and_code(stripped)
    if is_markup:
        return False, reason
        
    # Repetition & Spam check
    is_spam, reason = check_repetition_and_spam(stripped)
    if is_spam:
        return False, reason
        
    # Excessive non-word symbol noise (e.g. @#$%^&*_)
    # Note: Indic matras / vowel signs are Unicode category 'M*' (Mn/Mc), which are NOT symbols!
    # We define symbol noise as actual math/currency/modifier symbols and unusual punctuation
    # excluding standard punctuation and Devanagari danda
    allowed_chars = set(" ।॥.,!?-;:\"'()[]{}/\\")
    noise_symbols = sum(
        1 for c in stripped
        if not ('\u0900' <= c <= '\u097F')
        and not c.isalnum()
        and not c.isspace()
        and c not in allowed_chars
    )
    if noise_symbols / max(len(stripped), 1) > 0.20:
        return False, "excessive_symbol_noise"
        
    # Legitimate unusual short text (proverbs) or formatted verse/tables
    if is_legitimate_unusual_text(stripped):
        return True, None
        
    # Standard length floor for normal prose
    words = WORD_REGEX.findall(stripped)
    word_count = len(words)
    if len(stripped) < 15 or word_count < 3:
        return False, "too_short_insufficient_content"
        
    return True, None
