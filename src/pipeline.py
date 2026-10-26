"""
src/pipeline.py - Core Indic Training Corpus Cleaning and Curation Pipeline.

Executes streaming curation:
1. Streaming Ingestion: Fault-tolerant reading of raw files, tracking malformed records.
2. Normalization: Unicode NFKC, Devanagari nukta canonicalization, invisible character removal, whitespace.
3. Language ID: Tri-state classifier ('hi', 'hi-Latn', 'en or other').
4. Quality Filtering: Rejection of spam, boilerplate, markup, garbled text while preserving proverbs/verse/tables.
5. Exact & Near Deduplication: MD5 exact hash + MinHash LSH shingle Jaccard dedup.
6. Eval Set & Contamination Check: ~200 held-out items, checking for exact, normalized, near, and cross-script leaks.
7. PII Redaction: Replacing Indian mobile/landline phones, emails, Aadhaar, PAN with semantic tags.
8. Deterministic Output: Atomic writes for clean corpus, ledger, eval set, contamination report, and manifest.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import gzip
import time
import random
import hashlib
from typing import Dict, List, Set, Any, Optional

from src.reader import StreamingReader
from src.normalizer import normalize_text, normalize_for_comparison
from src.lang_id import identify_language
from src.filter import filter_quality
from src.dedup import Deduplicator
from src.pii import redact_pii
from src.contamination import ContaminationChecker, transliterate_devanagari_to_latin


class CorpusPipeline:
    def __init__(self, raw_dir: str, out_dir: str, near_dup_threshold: float = 0.80, eval_size: int = 200, seed: int = 42):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.near_dup_threshold = near_dup_threshold
        self.eval_size = eval_size
        self.seed = seed
        
        # Stages count tracking
        self.stage_counts = {
            "total_input_records": 0,
            "read_malformed": 0,
            "lang_hi": 0,
            "lang_hi_latn": 0,
            "lang_en_or_other_dropped": 0,
            "quality_filter_dropped": 0,
            "exact_duplicate_dropped": 0,
            "near_duplicate_dropped": 0,
            "eval_held_out": 0,
            "contamination_purged": 0,
            "pii_redacted_documents": 0,
            "clean_kept_records": 0
        }
        
        # Per-language funnel tracking
        # Each lang maps to dict of stage -> count
        self.lang_funnel = {
            "hi": {"input": 0, "quality_passed": 0, "exact_dedup_passed": 0, "near_dedup_passed": 0, "clean_kept": 0},
            "hi-Latn": {"input": 0, "quality_passed": 0, "exact_dedup_passed": 0, "near_dedup_passed": 0, "clean_kept": 0},
            "en or other": {"input": 0, "quality_passed": 0, "exact_dedup_passed": 0, "near_dedup_passed": 0, "clean_kept": 0}
        }
        
        self.pii_totals = {"phone": 0, "email": 0, "aadhaar": 0, "pan": 0}

    def run(self):
        start_time = time.time()
        print(f"=== Starting Indic Corpus Cleaning Pipeline ===")
        print(f"Raw Input Directory : {self.raw_dir}")
        print(f"Output Directory    : {self.out_dir}")
        print(f"Near Dedup Threshold: {self.near_dup_threshold}")
        print(f"Deterministic Seed  : {self.seed}")

        # Ensure output directories exist
        clean_dir = self.out_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Temporary files for atomic writes
        clean_tmp_path = clean_dir / "corpus.jsonl.gz.tmp"
        ledger_tmp_path = self.out_dir / "ledger.jsonl.gz.tmp"
        eval_path = self.out_dir / "eval.jsonl"
        contamination_path = self.out_dir / "contamination.jsonl"
        manifest_path = self.out_dir / "manifest.json"

        # Clean up any stale temp files from aborted runs
        for tmp_file in [clean_tmp_path, ledger_tmp_path]:
            if tmp_file.exists():
                tmp_file.unlink()

        # Initialize deduplicator
        deduplicator = Deduplicator(near_threshold=self.near_dup_threshold, track_near_pairs=True)

        # First pass: Stream records, perform read, normalize, lang_id, filter, and dedup
        reader = StreamingReader(str(self.raw_dir))
        
        # We will hold out an evaluation set of ~200 items (balanced between hi and hi-Latn)
        eval_items: List[Dict[str, Any]] = []
        eval_ids: Set[str] = set()
        target_eval_hi = int(self.eval_size * 0.60)     # 120 Devanagari Hindi
        target_eval_latn = int(self.eval_size * 0.40)   # 80 Hinglish
        
        # Buffer of candidate clean training records for contamination check
        clean_candidates: List[Dict[str, Any]] = []
        
        print("\n[Stage 1/3] Streaming raw records through Normalization, LangID, Quality Filter, & Dedup...", flush=True)
        
        with gzip.open(ledger_tmp_path, "wt", encoding="utf-8") as ledger_file:
            doc_idx = 0
            for record, is_valid, skip_reason in reader.iterate_records():
                doc_idx += 1
                self.stage_counts["total_input_records"] += 1
                doc_id = record["id"] if record else f"doc_unknown_{doc_idx}"
                source_name = record["source"] if record else "unknown"

                if doc_idx % 2500 == 0:
                    print(f"  Processed {doc_idx:,} records... (clean candidates: {len(clean_candidates):,})", flush=True)

                # Handle malformed / unreadable records from reader
                if not is_valid:
                    self.stage_counts["read_malformed"] += 1
                    ledger_row = {
                        "id": doc_id,
                        "status": "dropped",
                        "stage": "read",
                        "reason": skip_reason or "malformed_record",
                        "dup_of": None
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                    continue

                raw_text = record.get("text", "")

                # Normalization
                norm_text = normalize_text(raw_text)

                # Language Identification
                lang = identify_language(norm_text)
                self.lang_funnel[lang]["input"] += 1

                if lang == "hi":
                    self.stage_counts["lang_hi"] += 1
                elif lang == "hi-Latn":
                    self.stage_counts["lang_hi_latn"] += 1
                else:
                    self.stage_counts["lang_en_or_other_dropped"] += 1
                    ledger_row = {
                        "id": doc_id,
                        "status": "dropped",
                        "stage": "lang_id",
                        "reason": f"language_not_target ({lang})",
                        "dup_of": None
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                    continue

                # Quality Filter
                passes_quality, quality_reason = filter_quality(norm_text)
                if not passes_quality:
                    self.stage_counts["quality_filter_dropped"] += 1
                    ledger_row = {
                        "id": doc_id,
                        "status": "dropped",
                        "stage": "filter_quality",
                        "reason": quality_reason or "low_quality",
                        "dup_of": None
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                    continue

                self.lang_funnel[lang]["quality_passed"] += 1

                # Exact Deduplication
                comp_text = normalize_for_comparison(norm_text)
                exact_dup_of = deduplicator.check_exact(doc_id, comp_text)
                if exact_dup_of:
                    self.stage_counts["exact_duplicate_dropped"] += 1
                    ledger_row = {
                        "id": doc_id,
                        "status": "dropped",
                        "stage": "dedup_exact",
                        "reason": "exact_duplicate",
                        "dup_of": exact_dup_of
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                    continue

                self.lang_funnel[lang]["exact_dedup_passed"] += 1

                # Near Deduplication
                near_dup_of, near_sim = deduplicator.check_near(doc_id, comp_text)
                if near_dup_of:
                    self.stage_counts["near_duplicate_dropped"] += 1
                    ledger_row = {
                        "id": doc_id,
                        "status": "dropped",
                        "stage": "dedup_near",
                        "reason": f"near_duplicate (similarity={near_sim:.3f})",
                        "dup_of": near_dup_of
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                    continue

                self.lang_funnel[lang]["near_dedup_passed"] += 1

                # Hold-out selection for Eval Set
                # We select high-quality clean documents evenly sampled for the eval set
                is_eval_item = False
                if lang == "hi" and sum(1 for e in eval_items if e["lang"] == "hi") < target_eval_hi:
                    if doc_idx % 23 == 0:  # deterministic stride sampling
                        is_eval_item = True
                elif lang == "hi-Latn" and sum(1 for e in eval_items if e["lang"] == "hi-Latn") < target_eval_latn:
                    if doc_idx % 17 == 0:
                        is_eval_item = True

                if is_eval_item:
                    eval_items.append({
                        "id": doc_id,
                        "text": norm_text,
                        "lang": lang,
                        "source": source_name
                    })
                    eval_ids.add(doc_id)
                    self.stage_counts["eval_held_out"] += 1
                    ledger_row = {
                        "id": doc_id,
                        "status": "dropped",
                        "stage": "eval_holdout",
                        "reason": "held_out_for_eval_set",
                        "dup_of": None
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                    continue

                # Add to candidate clean pool
                clean_candidates.append({
                    "id": doc_id,
                    "text": norm_text,
                    "comp_text": comp_text,
                    "lang": lang,
                    "source": source_name
                })

            print(f"  Stage 1 completed. Total candidates for training: {len(clean_candidates):,}")
            print(f"  Held-out evaluation set items: {len(eval_items):,}")

            # Save Eval Set to <out_dir>/eval.jsonl
            print(f"\n[Stage 2/3] Writing Eval Set & Checking for Contamination Leaks...")
            eval_manifest = {
                "metric_description": "Measures multi-task Indic comprehension, fluency, and code-mixed Hindi-English reasoning across conversational and informational registers without train-set memorization.",
                "total_items": len(eval_items),
                "hi_count": sum(1 for e in eval_items if e["lang"] == "hi"),
                "hi_latn_count": sum(1 for e in eval_items if e["lang"] == "hi-Latn")
            }
            with open(eval_path, "w", encoding="utf-8") as eval_file:
                for item in eval_items:
                    eval_file.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Initialize Contamination Checker with held-out eval items
            contamination_checker = ContaminationChecker(eval_items)

            # Check training candidates for leaks
            contaminated_eval_map: Dict[str, List[str]] = {}
            contamination_method_map: Dict[str, str] = {}
            leaked_train_ids: Set[str] = set()

            for cand in clean_candidates:
                leak_res = contamination_checker.check_training_record(
                    train_id=cand["id"],
                    raw_text=cand["text"],
                    norm_text=cand["comp_text"],
                    lang=cand["lang"]
                )
                if leak_res:
                    leaked_eval_id, method = leak_res
                    leaked_train_ids.add(cand["id"])
                    if leaked_eval_id not in contaminated_eval_map:
                        contaminated_eval_map[leaked_eval_id] = []
                    contaminated_eval_map[leaked_eval_id].append(cand["id"])
                    contamination_method_map[leaked_eval_id] = method

            self.stage_counts["contamination_purged"] = len(leaked_train_ids)
            print(f"  Contamination check finished: {len(leaked_train_ids)} train documents purged across {len(contaminated_eval_map)} affected eval items.")

            # Write contamination.jsonl
            with open(contamination_path, "w", encoding="utf-8") as contam_file:
                for eval_id, train_id_list in contaminated_eval_map.items():
                    method = contamination_method_map.get(eval_id, "near_copy")
                    contam_file.write(json.dumps({
                        "eval_id": eval_id,
                        "train_ids": train_id_list,
                        "method": method
                    }, ensure_ascii=False) + "\n")

            # Record contamination purged rows in ledger
            for train_id in leaked_train_ids:
                ledger_row = {
                    "id": train_id,
                    "status": "dropped",
                    "stage": "contamination",
                    "reason": "eval_set_leakage_detected",
                    "dup_of": None
                }
                ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")

            # Stage 3: PII Redaction & Output Clean Corpus
            print(f"\n[Stage 3/3] Performing PII Redaction & Emitting Final Clean Corpus...")
            with gzip.open(clean_tmp_path, "wt", encoding="utf-8") as clean_file:
                for cand in clean_candidates:
                    if cand["id"] in leaked_train_ids:
                        continue

                    # Redact PII
                    redacted_text, pii_stats = redact_pii(cand["text"])
                    if pii_stats["total"] > 0:
                        self.stage_counts["pii_redacted_documents"] += 1
                        for pii_type in ("phone", "email", "aadhaar", "pan"):
                            self.pii_totals[pii_type] += pii_stats[pii_type]

                    # Clean row output
                    clean_record = {
                        "id": cand["id"],
                        "text": redacted_text,
                        "lang": cand["lang"],
                        "source": cand["source"]
                    }
                    clean_file.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                    
                    self.stage_counts["clean_kept_records"] += 1
                    self.lang_funnel[cand["lang"]]["clean_kept"] += 1

                    # Record 'kept' in ledger
                    ledger_row = {
                        "id": cand["id"],
                        "status": "kept",
                        "stage": "clean",
                        "reason": "passed_all_filters",
                        "dup_of": None
                    }
                    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")

        # Atomic replacement: rename tmp files to final files
        final_clean_path = clean_dir / "corpus.jsonl.gz"
        final_ledger_path = self.out_dir / "ledger.jsonl.gz"

        if final_clean_path.exists():
            final_clean_path.unlink()
        clean_tmp_path.rename(final_clean_path)

        if final_ledger_path.exists():
            final_ledger_path.unlink()
        ledger_tmp_path.rename(final_ledger_path)

        # Generate a 1,000-row random sample of cleaned output (required by "What to send")
        sample_path = self.out_dir / "clean_sample_1000.jsonl"
        self._generate_clean_sample(final_clean_path, sample_path, sample_size=1000)

        # Compute file hashes
        input_file_hashes = self._compute_input_hashes()

        # Build Manifest
        manifest = {
            "pipeline": "IndicCorpusCleaningPipeline",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - start_time, 2),
            "parameters": {
                "near_dup_threshold": self.near_dup_threshold,
                "minhash_permutations": 64,
                "lsh_bands": 16,
                "lsh_rows_per_band": 4,
                "shingle_k": 5,
                "eval_size": len(eval_items),
                "random_seed": self.seed
            },
            "eval_specification": eval_manifest,
            "input_file_hashes": input_file_hashes,
            "stage_counts": self.stage_counts,
            "language_funnel": self.lang_funnel,
            "pii_redaction_totals": self.pii_totals,
            "borderline_near_duplicate_pairs": deduplicator.inspected_pairs[:10]
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n=== Pipeline Completed Successfully in {time.time() - start_time:.2f}s ===")
        print(f"Total Input Records      : {self.stage_counts['total_input_records']:,}")
        print(f"Clean Output Documents   : {self.stage_counts['clean_kept_records']:,}")
        print(f"  - Hindi (Devanagari)   : {self.lang_funnel['hi']['clean_kept']:,}")
        print(f"  - Hinglish (hi-Latn)   : {self.lang_funnel['hi-Latn']['clean_kept']:,}")
        print(f"Eval Set Size            : {len(eval_items):,}")
        print(f"Contamination Purged     : {self.stage_counts['contamination_purged']:,}")
        print(f"Outputs generated at     : {self.out_dir}")

    def _generate_clean_sample(self, clean_gz_path: Path, sample_path: Path, sample_size: int = 1000):
        """Deterministically reservoir-sample 1,000 rows from the cleaned output."""
        random.seed(self.seed)
        reservoir = []
        with gzip.open(clean_gz_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < sample_size:
                    reservoir.append(line)
                else:
                    j = random.randint(0, i)
                    if j < sample_size:
                        reservoir[j] = line
                        
        with open(sample_path, "w", encoding="utf-8") as out:
            for line in reservoir:
                out.write(line)
        print(f"  Emitted 1,000-row sample to {sample_path}")

    def _compute_input_hashes(self) -> Dict[str, str]:
        """Compute sha256 hashes of all input files in raw_dir."""
        hashes = {}
        for p in sorted(self.raw_dir.glob("**/*")):
            if p.is_file() and not p.name.startswith("."):
                sha = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        sha.update(chunk)
                hashes[p.name] = sha.hexdigest()
        return hashes


def main():
    if len(sys.argv) < 3:
        print("Usage: python src/pipeline.py <raw_dir> <out_dir> [--threshold 0.80]")
        sys.exit(1)
        
    raw_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    threshold = 0.80
    if len(sys.argv) > 3:
        try:
            threshold = float(sys.argv[3])
        except ValueError:
            pass

    pipeline = CorpusPipeline(raw_dir, out_dir, near_dup_threshold=threshold)
    pipeline.run()


if __name__ == "__main__":
    main()
