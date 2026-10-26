"""
src/lang_id.py - Tri-state language classifier for Indic LLM corpus curation.

Labels every document with EXACTLY one of:
- 'hi'      : Hindi in Devanagari, including Devanagari text with English words mixed in
- 'hi-Latn' : Hindi written in Latin script, including code-mixed Hinglish
- 'en or other' : Pure English or any other language/script

Does not rely on external black-box models or untrusted source metadata.
Deterministic, fast, and robust across diverse domains.
"""

import re
from typing import Tuple, Dict

# Unicode ranges
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")
LATIN_REGEX = re.compile(r"[a-zA-Z]")
WORD_REGEX = re.compile(r"\b[\w\u0900-\u097F]+\b")

# High-frequency Devanagari Hindi markers (particles, pronouns, auxiliaries)
HINDI_DEVANAGARI_STOPWORDS = {
    "है", "हैं", "का", "की", "के", "में", "से", "को", "पर", "और", "नहीं", "यह", "वह",
    "था", "थी", "थे", "किया", "गया", "करने", "होता", "होती", "होते", "होने", "रहा",
    "रही", "रहे", "इस", "उस", "एक", "दो", "भी", "तो", "ही", "ने", "या", "कर", "दिया",
    "गए", "गई", "लिए", "अपने", "अपनी", "अपना", "सकते", "सकता", "सकती", "हुए", "हुआ",
    "हुई", "लेकिन", "कि", "जो", "तक", "साथ", "सब", "कोई", "कुछ", "क्या", "कहा"
}

# Strong Hinglish lexical markers that unambiguously indicate Hindi in Latin script
# (None of these overlap with standard English common dictionary words)
HINGLISH_STRONG_LEXICON = {
    # Pronouns & determiners
    "aap", "aapka", "aapki", "aapke", "aapko", "aapne",
    "hum", "humko", "humne", "hamara", "hamari", "hamare",
    "mai", "main", "mujhe", "mujhko", "mera", "meri", "mere", "maine",
    "tu", "tujhe", "tera", "teri", "tere", "tune",
    "tum", "tumhe", "tumhara", "tumhari", "tumhare", "tumne",
    "uska", "uski", "uske", "usko", "usne", "unka", "unki", "unke", "unhe", "unhone",
    "iska", "iski", "iske", "isko", "isne", "inka", "inki", "inke", "inhe", "inhone",
    "yeh", "ye", "woh", "wo", "yaha", "yahan", "waha", "wahan", "idhar", "udhar",
    "kya", "kaun", "kon", "kab", "kahan", "kaha", "kidhar", "kaise", "kese", "kyun", "kyu", "kyo",
    "kis", "kise", "kisko", "kisne", "kisi", "kisse",
    "kuch", "kuchh", "koi", "sab", "sabhi", "sabko", "sabka",
    
    # Auxiliaries & copulas
    "hai", "hain", "hoon", "hun", "hn",
    "tha", "thi", "thay",
    "hoga", "hogi", "hoge", "honge", "hona", "hone",
    "raha", "rahi", "rahe", "rha", "rhi", "rhe", "raho",
    
    # Common Hindi verbs (inflected / base)
    "kar", "karo", "kare", "karein", "karna", "karne", "karni", "karta", "karti", "karte",
    "kiya", "kiye", "kra", "kri", "krr",
    "dena", "dene", "deni", "deta", "deti", "dete", "diya", "diye",
    "lena", "lene", "leni", "leta", "leti", "lete", "liya", "liye",
    "bol", "bolo", "bole", "bolna", "bolne", "bola", "boli", "bolein",
    "bata", "btao", "batao", "bataye", "batana", "batane", "bataya", "batayi",
    "suno", "sune", "sunna", "suna",
    "dekh", "dekho", "dekhe", "dekhna", "dekhne", "dekha", "dekhi",
    "samajh", "samjho", "samjhe", "samajhna", "samjha", "samjhi",
    "chal", "chalo", "chale", "chalna", "chala", "chali", "chalte",
    "jao", "jae", "jaye", "jana", "jaana", "jaate", "jaati", "gaya", "gayi", "gaye", "gya", "gyi", "gye",
    "aao", "aaye", "aana", "aata", "aati", "aate", "aaya", "aayi",
    "milo", "mile", "milna", "mila", "mili", "milte",
    "socho", "soche", "sochna", "socha",
    "rakho", "rakhe", "rakhna", "rakha",
    "lagta", "lagti", "lagte", "lagna", "laga", "lagi", "lage",
    
    # Particles, adverbs, conjunctions
    "aur", "lekin", "magar", "parantu", "kintu",
    "kyunki", "kyuki", "kuki", "isliye", "esliye", "taaki", "taki",
    "toh", "bhi", "nahi", "nhi", "nahin", "mat",
    "abhi", "tabhi", "jabki", "kabhi", "hamesha",
    "pehle", "baad", "phir", "fir", "dobara",
    "mein", "saath", "paas", "andar", "bahar", "upar", "neeche",
    "bohot", "bahut", "bhot", "thoda", "thodi", "zyada", "jyada",
    "accha", "acchi", "acche", "achha", "achhi", "achhe", "bura", "buri", "bure",
    "sahi", "galat", "theek", "thik", "badhiya", "shukriya", "dhanyawad", "namaste",
    "bhai", "yaar", "dost", "logon", "baat", "baatein", "kaam", "waqt",
    "din", "raat", "saal", "mahina", "kal", "aaj", "parso", "waise", "vaise", "aise", "jaise",
    "pata", "maloom", "zaroor", "pakka", "sach"
}

# Ambiguous words: common in Hindi transliterations, but also valid English words/abbreviations.
# These only count toward Hinglish if at least one strong Hinglish marker is also present.
HINGLISH_AMBIGUOUS_LEXICON = {
    "to", "is", "in", "us", "me", "so", "no", "he", "be", "do", "or", "par", "se", "ko", "ke", "ki", "ka", "pe", "the"
}

# Pure English common stopwords for comparison
ENGLISH_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "are", "is", "was", "were", "been"
}


def compute_script_counts(text: str) -> Dict[str, int]:
    """Compute character counts by script type."""
    devanagari_count = len(DEVANAGARI_REGEX.findall(text))
    latin_count = len(LATIN_REGEX.findall(text))
    total_letters = devanagari_count + latin_count
    
    return {
        "devanagari": devanagari_count,
        "latin": latin_count,
        "total_letters": total_letters
    }


def identify_language(text: str) -> str:
    """
    Classify document language strictly into one of:
    - 'hi'
    - 'hi-Latn'
    - 'en or other'
    """
    if not text or not text.strip():
        return "en or other"
        
    counts = compute_script_counts(text)
    total_letters = counts["total_letters"]
    
    if total_letters == 0:
        return "en or other"
        
    dev_ratio = counts["devanagari"] / total_letters
    lat_ratio = counts["latin"] / total_letters
    
    # -------------------------------------------------------------
    # Case 1: Hindi in Devanagari ('hi')
    # Meets the requirement: "Hindi in Devanagari, including Devanagari
    # text with English words mixed in".
    # -------------------------------------------------------------
    if counts["devanagari"] >= 10 and dev_ratio >= 0.20:
        # Check presence of Devanagari characters and Hindi markers
        tokens = set(WORD_REGEX.findall(text))
        hindi_markers_found = tokens.intersection(HINDI_DEVANAGARI_STOPWORDS)
        # If there are explicit Hindi markers or dominant Devanagari script (> 40%)
        if len(hindi_markers_found) >= 1 or dev_ratio >= 0.40:
            return "hi"

    # Even with shorter text, if Devanagari dominates
    if dev_ratio >= 0.60:
        return "hi"
        
    # -------------------------------------------------------------
    # Case 2: Hinglish in Latin Script ('hi-Latn')
    # Hindi written in Latin script, including code-mixed Hinglish
    # -------------------------------------------------------------
    if lat_ratio >= 0.50:
        raw_tokens = WORD_REGEX.findall(text.lower())
        total_tokens = len(raw_tokens)
        
        if total_tokens == 0:
            return "en or other"
            
        strong_matches = [t for t in raw_tokens if t in HINGLISH_STRONG_LEXICON]
        unique_strong = set(strong_matches)
        ambiguous_matches = [t for t in raw_tokens if t in HINGLISH_AMBIGUOUS_LEXICON]
        english_matches = [t for t in raw_tokens if t in ENGLISH_STOPWORDS]
        
        strong_count = len(strong_matches)
        ambiguous_count = len(ambiguous_matches)
        english_count = len(english_matches)
        
        is_hinglish = False
        
        # High confidence Hinglish indicators:
        # 1. At least 2 distinct strong Hinglish words
        if len(unique_strong) >= 2:
            is_hinglish = True
        # 2. At least 1 strong Hinglish word with high density or ambiguous Hindi context
        elif len(unique_strong) >= 1 and (strong_count / total_tokens >= 0.08 or ambiguous_count >= 1):
            is_hinglish = True
        # 3. Code-mixed with at least one strong word and not overwhelmed by standard English
        elif len(unique_strong) >= 1 and (strong_count + ambiguous_count >= english_count * 0.5):
            is_hinglish = True
            
        if is_hinglish:
            return "hi-Latn"
            
    # -------------------------------------------------------------
    # Case 3: 'en or other'
    # Pure English, other languages (Spanish, German, etc.), or non-Hindi scripts
    # -------------------------------------------------------------
    return "en or other"
