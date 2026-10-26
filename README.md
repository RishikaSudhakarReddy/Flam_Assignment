# Indic Corpus Cleaner

A robust, streaming data-curation pipeline designed to collect, clean, filter, and deduplicate Hindi (Devanagari) and Hinglish (Latin script) text corpora for training large language models (such as Falcon), accompanied by a provably uncontaminated evaluation set.

## Overview

High-quality Indic LLM training requires addressing distinct linguistic challenges beyond ASCII:
- Complex Devanagari Unicode equivalence (NFC/NFKC, composite Nukta forms, stray halants/matras, and zero-width format characters ZWJ/ZWNJ/ZWSP).
- Script-aware tri-state language identification (`hi`, `hi-Latn`, `en or other`) that does not trust external metadata labels.
- Deliberate quality filtering that purges spam, web boilerplate, and encoding artifacts (mojibake) while explicitly protecting legitimate unusual texts such as short proverbs, poetry, and numeric tables/scorecards.
- Streaming exact and near-deduplication (character 5-gram MinHash with Locality-Sensitive Hashing).
- Context-sensitive Indian PII redaction (`<PHONE>`, `<EMAIL>`, `<AADHAAR>`, `<PAN>`) that strictly protects non-PII numbers (years, Indian pin codes, currency amounts, match scores).
- Rigorous cross-script contamination detection to prevent training data leaks into held-out evaluation benchmarks.
- Streaming, deterministic, restart-safe offline execution.

---

## Architecture & Directory Layout

```
indic-corpus-cleaner/
├── download.sh            # Downloads raw corpora with SHA-256 validation
├── download.py            # Python downloader with metadata tracking
├── run.sh                 # Offline pipeline entry point (POSIX shell)
├── requirements.txt       # Minimal dependency specification (pyarrow)
├── README.md              # Project architecture and execution guide
├── DECISIONS.md           # Engineering decisions, metrics, and trade-off write-up
├── src/
│   ├── __init__.py
│   ├── reader.py          # Fault-tolerant streaming raw input reader
│   ├── normalizer.py      # Unicode NFKC, Devanagari nukta, & invisible char cleaner
│   ├── lang_id.py         # Tri-state language classifier ('hi', 'hi-Latn', 'en or other')
│   ├── filter.py          # Quality, spam, and boilerplate filter with proverbs/verse preservation
│   ├── dedup.py           # Streaming exact (MD5) and near (MinHash LSH) deduplicator
│   ├── pii.py             # Indian PII redactor (Phone, Email, Aadhaar Verhoeff, PAN)
│   ├── contamination.py   # Eval holdout & cross-script contamination detector
│   └── pipeline.py        # Pipeline orchestrator and atomic writer
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   # Comprehensive unit and integration test suite
└── data/
    ├── raw/               # Downloaded raw input datasets
    └── out/               # Curation outputs
        ├── clean/         # Cleaned training corpus (*.jsonl.gz)
        ├── ledger.jsonl.gz# Audit ledger tracking every input document
        ├── eval.jsonl     # Held-out 200-item evaluation set
        ├── contamination.jsonl # Contamination leakage audit report
        └── manifest.json  # Hashes, thresholds, run metadata, and funnel counts
```

---

## Quickstart

### 1. Requirements
- Python 3.10+
- `pyarrow>=14.0.0`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Raw Corpora
Downloading is kept strictly decoupled in `download.sh`:
```bash
bash download.sh data/raw
```
This fetches raw files from three public sources:
1. `aaditya/databricks-dolly-15k-Hindi` (CC BY-SA 3.0) — 15,010 Hindi documents in Devanagari script.
2. `DSMJ910/hinglish-instruct-10k` (Apache-2.0) — 10,594 Roman script Hinglish documents.
3. `l3cube-pune/code-mixed-nlp` (`L3Cube-HingLID`) (CC BY-NC-SA 4.0) — 31,756 Roman Hinglish social media utterances.

### 3. Run the Offline Pipeline
`run.sh` operates 100% offline with zero network calls:
```bash
./run.sh data/raw data/out
```
Or on Windows:
```powershell
python src/pipeline.py data/raw data/out
```

---

## Output Specifications

| Path | Format | Description |
| :--- | :--- | :--- |
| `<out_dir>/clean/` | `corpus.jsonl.gz` | Kept documents with fields: `id`, `text`, `lang`, `source` |
| `<out_dir>/ledger.*` | `ledger.jsonl.gz` | One row per input document: `id`, `status` (`kept`/`dropped`), `stage`, `reason`, `dup_of` |
| `<out_dir>/eval.jsonl` | `eval.jsonl` | Held-out 200-item evaluation set balanced between Hindi and Hinglish |
| `<out_dir>/contamination.jsonl` | `contamination.jsonl` | Leaked eval records: `eval_id`, `train_ids`, `method` |
| `<out_dir>/manifest.json` | `manifest.json` | Input hashes, stage counts, funnel statistics, and parameters |

---

## Running Unit Tests

Run the test suite to verify normalizer, language classifier, quality filters, Verhoeff Aadhaar validation, near-deduplication, and cross-script transliteration:
```bash
python -m unittest tests/test_pipeline.py
```
