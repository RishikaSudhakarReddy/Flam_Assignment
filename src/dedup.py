"""
src/dedup.py - Streaming Exact and Near-Duplicate Deduplicator.

Features:
- Exact deduplication via 128-bit cryptographic hash of normalized text
- Near-deduplication via Character 5-gram MinHash with Locality Sensitive Hashing (LSH)
- Streaming ledger recording (identifying first occurrence doc_id as dup_of)
- Diagnostic inspection of pairs close to the similarity threshold
"""

import re
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any

# Number of MinHash permutations (K = 32)
NUM_PERMUTATIONS = 32
# Number of LSH bands (B = 8) and rows per band (R = 4) -> B * R = 32
NUM_BANDS = 8
ROWS_PER_BAND = 4

# Prime modulus for linear congruential hash functions: (a * x + b) % M
MERSENNE_PRIME = (1 << 61) - 1

# Deterministic coefficients for MinHash permutations (reproducible seed)
def _generate_hash_seeds(k: int) -> Tuple[List[int], List[int]]:
    import random
    rng = random.Random(42)
    a = [rng.randint(1, MERSENNE_PRIME - 1) for _ in range(k)]
    b = [rng.randint(0, MERSENNE_PRIME - 1) for _ in range(k)]
    return a, b

A_COEFFS, B_COEFFS = _generate_hash_seeds(NUM_PERMUTATIONS)


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """Generate character k-grams (shingles) from text."""
    compact = re.sub(r"\s+", " ", text.lower().strip())
    if len(compact) < k:
        return {compact}
    return {compact[i:i+k] for i in range(len(compact) - k + 1)}


def compute_minhash_signature(shingles: Set[str]) -> List[int]:
    """Compute K-dimensional MinHash signature vector for a set of shingles."""
    if not shingles:
        return [0] * NUM_PERMUTATIONS
        
    hashed_shingles = [
        int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], byteorder="big")
        for s in shingles
    ]
    
    signature = []
    for i in range(NUM_PERMUTATIONS):
        a = A_COEFFS[i]
        b = B_COEFFS[i]
        min_val = min((a * h + b) % MERSENNE_PRIME for h in hashed_shingles)
        signature.append(min_val)
        
    return signature


def estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for v1, v2 in zip(sig1, sig2) if v1 == v2)
    return matches / len(sig1)


def exact_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Compute true Jaccard similarity between two shingle sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


class Deduplicator:
    """
    Streaming Deduplicator for exact and near-duplicates.
    Maintains O(1) hash table for exact deduplication and banded LSH index
    for candidate retrieval during near-duplicate checking.
    """
    def __init__(self, near_threshold: float = 0.80, track_near_pairs: bool = True):
        self.near_threshold = near_threshold
        self.track_near_pairs = track_near_pairs
        
        # Exact deduplication: exact_hash -> first_doc_id
        self.exact_hashes: Dict[str, str] = {}
        
        # LSH buckets: band_idx -> {bucket_hash -> list of doc_ids}
        self.lsh_buckets: List[Dict[int, List[str]]] = [{} for _ in range(NUM_BANDS)]
        
        # In-memory storage for candidate comparison
        self.doc_signatures: Dict[str, List[int]] = {}
        self.doc_shingles: Dict[str, Set[str]] = {}
        self.doc_texts: Dict[str, str] = {}
        
        # Borderline candidate pairs recorded for threshold analysis in DECISIONS.md
        self.inspected_pairs: List[Dict[str, Any]] = []

    def check_exact(self, doc_id: str, normalized_text: str) -> Optional[str]:
        """
        Check if document is an exact duplicate.
        Returns first doc_id if duplicate, else None.
        """
        doc_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
        if doc_hash in self.exact_hashes:
            return self.exact_hashes[doc_hash]
        self.exact_hashes[doc_hash] = doc_id
        return None

    def check_near(self, doc_id: str, normalized_text: str) -> Tuple[Optional[str], float]:
        """
        Check if document is a near-duplicate against previously indexed documents.
        Returns (dup_of_id, similarity) if similarity >= threshold, else (None, best_similarity).
        """
        shingles = get_shingles(normalized_text, k=5)
        signature = compute_minhash_signature(shingles)
        
        # Query LSH candidate documents across all bands
        candidate_counts: Dict[str, int] = {}
        for b in range(NUM_BANDS):
            start = b * ROWS_PER_BAND
            band_chunk = tuple(signature[start : start + ROWS_PER_BAND])
            bucket_hash = hash(band_chunk)
            
            bucket = self.lsh_buckets[b].get(bucket_hash, [])
            for cand_id in bucket:
                candidate_counts[cand_id] = candidate_counts.get(cand_id, 0) + 1
                
        # Sort candidates by number of shared bands (most likely first) and evaluate top candidates
        sorted_candidates = sorted(candidate_counts.keys(), key=lambda c: candidate_counts[c], reverse=True)[:30]
        
        best_match_id = None
        best_similarity = 0.0
        
        for cand_id in sorted_candidates:
            cand_sig = self.doc_signatures[cand_id]
            sig_sim = estimate_jaccard(signature, cand_sig)
            
            # If signature similarity is promising, compute exact shingle Jaccard
            if sig_sim >= (self.near_threshold - 0.15):
                cand_shingles = self.doc_shingles.get(cand_id)
                true_sim = exact_jaccard(shingles, cand_shingles) if cand_shingles else sig_sim
                    
                if true_sim > best_similarity:
                    best_similarity = true_sim
                    best_match_id = cand_id
                    
                # Track pairs close to threshold (0.65 to 0.95) for DECISIONS.md
                if self.track_near_pairs and 0.65 <= true_sim <= 0.95:
                    if len(self.inspected_pairs) < 50:
                        self.inspected_pairs.append({
                            "doc1_id": cand_id,
                            "doc2_id": doc_id,
                            "doc1_text": self.doc_texts.get(cand_id, "")[:150],
                            "doc2_text": normalized_text[:150],
                            "similarity": round(true_sim, 4),
                            "action": "DROPPED" if true_sim >= self.near_threshold else "KEPT"
                        })

        if best_similarity >= self.near_threshold and best_match_id:
            return best_match_id, best_similarity
            
        # If not a near duplicate, index this document
        self._index_doc(doc_id, signature, shingles, normalized_text)
        return None, best_similarity

    def _index_doc(self, doc_id: str, signature: List[int], shingles: Set[str], text: str):
        """Index document in LSH bands."""
        self.doc_signatures[doc_id] = signature
        self.doc_shingles[doc_id] = shingles
        if self.track_near_pairs:
            self.doc_texts[doc_id] = text[:200]
            
        for b in range(NUM_BANDS):
            start = b * ROWS_PER_BAND
            band_chunk = tuple(signature[start : start + ROWS_PER_BAND])
            bucket_hash = hash(band_chunk)
            if bucket_hash not in self.lsh_buckets[b]:
                self.lsh_buckets[b][bucket_hash] = []
            self.lsh_buckets[b][bucket_hash].append(doc_id)
