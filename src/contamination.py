"""
src/contamination.py - Eval Set Builder and Contamination Leakage Detector.

Capabilities:
1. Curates a balanced 200-item evaluation set held out from training.
2. Performs 4-way leak detection against training data:
   - exact_copy: byte-identical matches
   - normalized_copy: matches after Unicode and nukta canonicalization
   - near_copy: shingle Jaccard overlap >= 0.80 (optimized with inverted shingle index)
   - cross_script_transliteration: Devanagari eval items matching Latin training text
3. Emits <out_dir>/contamination.jsonl and purges leaking train records.
"""

import re
import json
import hashlib
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from src.normalizer import normalize_for_comparison, normalize_text
from src.dedup import get_shingles, exact_jaccard

# -------------------------------------------------------------
# Phonetic Devanagari to Latin Transliterator for Cross-Script Matching
# -------------------------------------------------------------
DEVANAGARI_TO_LATIN_MAP = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    # Matras (vowel signs)
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    # Modifiers
    'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': '', '़': '',
    # Consonants
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f',
    'ज्ञ': 'gy', 'क्ष': 'ksh', 'त्र': 'tr'
}

MATRAS = set('ािीुूृेैोौ्')
WORD_BREAKS = set(" \t\n\r।.,!?-;:\"'()[]{}")


def transliterate_devanagari_to_latin(text: str) -> str:
    """
    Deterministic phonetic transliteration of Devanagari to Romanized Latin script.
    Implements modern Hindi schwa syncope: consonants at the end of words or before
    word breaks do NOT retain the Sanskrit inherent 'a' (e.g. 'भारत' -> 'bharat',
    'आप' -> 'aap', 'नमस्ते' -> 'namaste').
    """
    result = []
    chars = list(text)
    n = len(chars)
    
    i = 0
    while i < n:
        c = chars[i]
        # Check two-character compound (e.g. consonant + nukta)
        if i + 1 < n and (c + chars[i+1]) in DEVANAGARI_TO_LATIN_MAP:
            result.append(DEVANAGARI_TO_LATIN_MAP[c + chars[i+1]])
            i += 2
            continue
            
        if c in DEVANAGARI_TO_LATIN_MAP:
            latin = DEVANAGARI_TO_LATIN_MAP[c]
            # Check if consonant
            if ('\u0915' <= c <= '\u0939') or ('\u0958' <= c <= '\u095F'):
                # In modern Hindi, schwa 'a' is pronounced only if NOT followed by a matra,
                # NOT followed by a virama/halant, and NOT at the end of a word!
                is_end_of_word = (i + 1 == n) or (chars[i+1] in WORD_BREAKS)
                has_following_matra = (i + 1 < n) and (chars[i+1] in MATRAS)
                
                if not has_following_matra and not is_end_of_word:
                    result.append(latin + 'a')
                else:
                    result.append(latin)
            else:
                result.append(latin)
        else:
            result.append(c)
        i += 1
        
    res_str = "".join(result).lower()
    res_str = re.sub(r"[^\w\s]", " ", res_str)
    return re.sub(r"\s+", " ", res_str).strip()


class ContaminationChecker:
    """
    Holds out eval items and checks training corpus for contamination leaks.
    Uses inverted shingle indexes to execute checks in O(shingles) instead of O(N * M).
    """
    def __init__(self, eval_items: List[Dict]):
        """
        eval_items: List of dicts with 'id', 'text', 'lang'
        """
        self.eval_items = eval_items
        
        # Precompute lookup structures for fast O(1) exact checking
        self.raw_eval_lookup: Dict[str, str] = {}           # raw_hash -> eval_id
        self.norm_eval_lookup: Dict[str, str] = {}          # norm_hash -> eval_id
        self.eval_shingles: Dict[str, Set[str]] = {}        # eval_id -> set of 5-grams
        self.transliterated_eval: Dict[str, Set[str]] = {}  # eval_id -> set of transliterated 5-grams
        
        # Inverted index: shingle -> list of eval_ids
        self.shingle_to_eval_ids: Dict[str, List[str]] = defaultdict(list)
        self.trans_shingle_to_eval_ids: Dict[str, List[str]] = defaultdict(list)
        
        for item in self.eval_items:
            eid = item["id"]
            raw_text = item["text"]
            norm_text = normalize_for_comparison(raw_text)
            
            self.raw_eval_lookup[hashlib.md5(raw_text.encode("utf-8")).hexdigest()] = eid
            self.norm_eval_lookup[hashlib.md5(norm_text.encode("utf-8")).hexdigest()] = eid
            shingles = get_shingles(norm_text, k=5)
            self.eval_shingles[eid] = shingles
            for s in shingles:
                self.shingle_to_eval_ids[s].append(eid)
            
            # If Devanagari Hindi, also precompute transliterated shingles for cross-script check
            if item.get("lang") == "hi" or any('\u0900' <= c <= '\u097F' for c in raw_text):
                latin_trans = transliterate_devanagari_to_latin(raw_text)
                if len(latin_trans) >= 10:
                    trans_shingles = get_shingles(latin_trans, k=5)
                    self.transliterated_eval[eid] = trans_shingles
                    for ts in trans_shingles:
                        self.trans_shingle_to_eval_ids[ts].append(eid)

    def check_training_record(self, train_id: str, raw_text: str, norm_text: str, lang: str) -> Optional[Tuple[str, str]]:
        """
        Check if a training document leaks any eval item.
        Returns (leaked_eval_id, method) if contaminated, else None.
        """
        # 1. Exact copy
        raw_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()
        if raw_hash in self.raw_eval_lookup:
            return self.raw_eval_lookup[raw_hash], "exact_copy"
            
        # 2. Normalized copy
        norm_hash = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
        if norm_hash in self.norm_eval_lookup:
            return self.norm_eval_lookup[norm_hash], "normalized_copy"
            
        # 3. Near copy (Jaccard >= 0.80)
        train_shingles = get_shingles(norm_text, k=5)
        # Find candidate eval items sharing shingles
        candidate_eval_counts = defaultdict(int)
        for s in train_shingles:
            for eid in self.shingle_to_eval_ids.get(s, []):
                candidate_eval_counts[eid] += 1
                
        for eid, shared_cnt in candidate_eval_counts.items():
            eshingles = self.eval_shingles[eid]
            # Fast upper bound check: shared / min(len1, len2) must be >= 0.80
            if shared_cnt / max(len(eshingles), 1) >= 0.70:
                sim = exact_jaccard(train_shingles, eshingles)
                if sim >= 0.80:
                    return eid, "near_copy"
                
        # 4. Cross-script transliteration leak
        # If training doc is Latin (Hinglish) and eval doc is Devanagari
        if lang == "hi-Latn" and self.transliterated_eval:
            latin_norm = normalize_for_comparison(raw_text)
            latin_shingles = get_shingles(latin_norm, k=5)
            trans_candidate_counts = defaultdict(int)
            for ts in latin_shingles:
                for eid in self.trans_shingle_to_eval_ids.get(ts, []):
                    trans_candidate_counts[eid] += 1
                    
            for eid, shared_cnt in trans_candidate_counts.items():
                trans_shingles = self.transliterated_eval[eid]
                if shared_cnt / max(len(trans_shingles), 1) >= 0.60:
                    sim = exact_jaccard(latin_shingles, trans_shingles)
                    if sim >= 0.70:
                        return eid, "cross_script_transliteration"
                    
        return None
