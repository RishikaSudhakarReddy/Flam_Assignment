# Data Engineering Decisions & Corpus Curation Write-up

This document details the quantitative outcomes, architectural decisions, and trade-offs from our end-to-end Indic training corpus pipeline run for **Falcon LLM**, targeting high-quality Hindi (Devanagari) and Hinglish (Roman script) pre-training data.

---

### 1. Sources

We collected 57,360 raw documents across three verified public sources, ensuring permissive licenses, diverse registers (instructional, conversational, social), and exact URL tracking:

* **Source 1: `aaditya/databricks-dolly-15k-Hindi`**
  * **License:** Creative Commons Attribution-ShareAlike 3.0 Unported (`CC BY-SA 3.0`)
  * **URL:** `https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hindi`
  * **Document Count:** 15,010 documents (Devanagari Hindi)
  * **Size:** 17.98 MB (compressed parquet), 38.52 MB uncompressed
  * **SHA-256:** `c052e622bed539eaf0d251334017b51fe296ac255b7ade815b432efdf049dd61`
* **Source 2: `DSMJ910/hinglish-instruct-10k`**
  * **License:** Apache License 2.0 (`Apache-2.0`)
  * **URL:** `https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k`
  * **Document Count:** 10,594 documents (100% Roman Hinglish)
  * **Size:** 3.01 MB (compressed parquet), 6.42 MB uncompressed
  * **SHA-256:** `8b0ddd04d5851a82c28e63406ec3dff3ae6a16ab3e2e5e3965c12116e0f1d787`
* **Source 3: `l3cube-pune/code-mixed-nlp` (`L3Cube-HingLID`)**
  * **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (`CC BY-NC-SA 4.0`)
  * **URL:** `https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/train.txt`
  * **Document Count:** 31,756 utterances (reconstructed from 999,987 token-level CoNLL tags)
  * **Size:** 7.42 MB (plain text)
  * **SHA-256:** `079607dd52d5cf56039b8ac6060fed9518914aa6b0e3d7699a799818d6b6574b`

*Total Raw Inputs:* **57,360 records** (28.41 MB raw on disk).

---

### 2. Funnel: Counts at Each Stage Per Language

Every record was streamed through our tri-state language identifier (`hi`, `hi-Latn`, `en or other`), quality filters, exact deduplicator, near deduplicator, and eval holdout. All transitions are audited in `ledger.jsonl.gz`:

| Pipeline Stage | Hindi (`hi`) | Hinglish (`hi-Latn`) | English / Other (`en or other`) | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Input Ingested** | **14,901** | **41,120** | **1,339** | **57,360** |
| *Language Filtered (Dropped non-Indic)* | — | — | 1,339 dropped | -1,339 |
| **Passed Quality Filters** | **14,885** (99.89%) | **40,958** (99.61%) | 0 | 55,843 |
| *Quality Filter Dropped* | 16 | 162 | — | -178 |
| **Passed Exact Deduplication** | **14,843** (99.72%) | **40,606** (99.14%) | 0 | 55,449 |
| *Exact Duplicate Dropped* | 42 | 352 | — | -394 |
| **Passed Near Deduplication** | **14,726** (99.21%) | **40,374** (99.43%) | 0 | 55,100 |
| *Near Duplicate Dropped* | 117 | 232 | — | -349 |
| *Held Out for Eval Set* | 120 | 80 | — | -200 |
| *Contamination Purged* | 0 | 0 | — | 0 |
| **Final Clean Output Kept** | **14,606** | **40,294** | **0** | **54,900** |

*Overall Yield:* **95.71%** retention of valid Indic records (54,900 clean training records + 200 evaluation items). Additionally, **98 documents** had sensitive Indian PII redacted (108 mobile/landline numbers, 13 email addresses, and 4 verified Aadhaar numbers).

---

### 3. Near-Duplicates: Method, Threshold, and Borderline Pairs

* **Method:** Character 5-gram shingling (`k=5`) with 32-permutation MinHash and 8 Locality-Sensitive Hashing (LSH) bands (4 rows per band). Candidate pairs sharing at least one band are retrieved and evaluated for true shingle Jaccard similarity $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$.
* **Threshold Selection ($0.80$):** We empirically benchmarked pairs between $0.65$ and $0.95$. Below $0.75$, distinct queries sharing general instruction templates (e.g. "निम्नलिखित पैराग्राफ को पढ़ें...") were falsely dropped. Above $0.85$, near-identical copypasta with minor punctuation or salutation tweaks leaked through. $0.80$ cleanly separates templated redundancy from distinct semantic intent.

**Three Borderline Pairs from our Run:**

1. **`dolly_15k_hindi#row_1207` vs `dolly_15k_hindi#row_2268` (Similarity: 0.8297) — Call: DROPPED**
   * *Doc 1:* `"कंप्यूटर साइंस में cap प्रमेय क्या है सैद्धांतिक कंप्यूटर विज्ञान में cap प्रमेय जिसे कंप्यूटर वैज्ञानिक एरिक ब्रूअर के नाम पर..."`
   * *Doc 2:* `"कंप्यूटर साइंस में सीएपी प्रमेय क्या है सैद्धांतिक कंप्यूटर विज्ञान में सीएपी प्रमेय जिसे कंप्यूटर वैज्ञानिक एरिक ब्रूअर के नाम पर..."`
   * *Analysis:* Both discuss Brewer's CAP Theorem with identical explanations. The only divergence is Latin `"cap"` vs Devanagari transliteration `"सीएपी"`. This is a classic near-duplicate; dropping it prevents model over-memorization.
2. **`dolly_15k_hindi#row_1623` vs `dolly_15k_hindi#row_2043` (Similarity: 0.7999) — Call: KEPT**
   * *Doc 1:* `"bernoulli s principle ke anusaar kinetic energy potential energy aur internal energy ka sum kya hona chahiye..."`
   * *Doc 2:* `"bernoulli s principle ko neeche diye gaye paragraphs ke based pe summarize karein..."`
   * *Analysis:* Both share the foundational physics background on fluid dynamics, but Doc 1 tests mathematical equation knowledge while Doc 2 evaluates summarization. Sitting just below the $0.80$ threshold, keeping both preserves complementary task diversity.
3. **`dolly_15k_hindi#row_1130` vs `dolly_15k_hindi#row_1140` (Similarity: 0.7497) — Call: KEPT**
   * *Doc 1:* `"कैबरनेट सॉविनन क्या है कैबरनेट सॉविनन फ्रेंच kabɛʁnɛ soviɲɔ दुनिया में सबसे ज्यादा जानी जाने वाली रेड वाइन..."`
   * *Doc 2:* `"कैबरनेट सॉविनन कहाँ उगाया जाता है कैबरनेट सॉविनन फ्रेंच kabɛʁnɛ soviɲɔ दुनिया भर में सबसे ज्यादा जाने जाने वाली..."`
   * *Analysis:* Doc 1 asks for the definition/type of Cabernet Sauvignon, whereas Doc 2 queries geographic cultivation regions. While context overlap is high (0.75), the target knowledge differs fundamentally; keeping both is the correct call.

---

### 4. Filters: Analysis of False Positives

Our filters deliberately whitelist short proverbs, rhymed poetry, and cricket/numeric scorecards. However, aggressive automated filtering produced two interesting borderline removals:

1. **`dolly_15k_hindi#row_3157` (Reason: `markup_excessive_html`)**
   * *Text:* `SELECT <कॉलम> FROM <table> WHERE <कॉलम> == <कंडीशन>`
   * *Issue:* The user asked an SQL query in Hindi. The output contained placeholder syntax with angle brackets (`<कॉलम>`, `<table>`, `<कंडीशन>`), which triggered the HTML regex (`<[^>]+>`) at 18.2% tag density.
   * *Resolution for Next Iteration:* Whitelist common programming language keywords (`SELECT`, `FROM`, `WHERE`) before applying HTML markup rejection.
2. **`dolly_15k_hindi#row_3488` (Reason: `spam_char_repetition`)**
   * *Text:* Contains driver log snippet `+++++ एंटर +++++`
   * *Issue:* The document demonstrated technical ODBC log tracing. The sequence `+++++` matched the 5+ repeated character threshold (`(.)\1{4,}`).
   * *Resolution for Next Iteration:* Restrict character repetition filtering to letters and emojis, while excluding common technical formatting delimiters (`+`, `-`, `=`).
3. **`dolly_15k_hindi#row_2123` (Reason: `garbled_mojibake_replacement_chars`)**
   * *Text:* Contains math notation for Burnside's Lemma necklace counting where an unmapped symbol became `\ufffd`. While the mathematical explanation was partially readable, dropping degraded formulas prevents hallucinated latex generation in Falcon.

---

### 5. Contamination: Methodology & Candidate Rejections

* **Method:** 4-tier check against our 200 held-out evaluation items:
  1. Exact string hash (MD5).
  2. Normalized hash (NFKC, nukta canonicalization, whitespace collapsed).
  3. Near-copy shingle Jaccard overlap $\ge 0.80$ (accelerated via inverted shingle index).
  4. Cross-script phonetic transliteration: Devanagari eval items are transliterated to Roman script applying modern Hindi schwa syncope (e.g. `भारत` $\to$ `bharat`, not `bharata`) and compared against Roman Hinglish training text with a $0.70$ Jaccard threshold.
* **Results:** **0 eval items contaminated**. Because our eval set was held out via deterministic stride sampling *after* deduplication, the remaining training pool remained provably disjoint.
* **Candidate Match Inspected & Rejected:**
  * *Eval item:* `eval_023` (`"ये बताओ कि ये राज्य और प्रांत कनाडा में हैं या यूएसए में: अलबामा, ओंटारियो..."`)
  * *Training candidate:* A geography document discussing Canadian provinces (`"कनाडा के प्रांतों की सूची..."`).
  * *Score:* Shingle overlap reached $0.48$. We rejected this match because while geographic entity vocabulary was shared, the factual structure and conversational instruction were completely independent. Setting the threshold at $0.70$ avoided false purging of general domain knowledge.

---

### 6. Reproducibility: Rebuilding Exactly in 6 Months

To reproduce this dataset byte-for-byte:
1. **Environment:** Python 3.12+ with `pip install pyarrow==26.0.0`.
2. **Download:** Execute `bash download.sh data/raw`. This pulls raw parquet and txt assets from the pinned Hugging Face and GitHub commit snapshots, verifying against the SHA-256 hashes recorded in `data/raw/sources_metadata.json`.
3. **Execute Pipeline:** Run `./run.sh data/raw data/out`.
4. **Determinism:** Seed `42` is fixed; filesystem directory iteration is lexicographically sorted; MinHash permutations use Mersenne prime modulus $(2^{61}-1)$ with constant RNG seeds; and atomic file renames prevent half-written artifacts.

---

### 7. Scale: Bottlenecks at 2 TB on Object Storage

If scaling this pipeline to 2 TB on object storage (e.g., S3/GCS with 20,000 parquet shards), our current single-node design would fail first at the **in-memory LSH deduplication state**: holding MinHash signatures and shingle sets for $\sim 2 \times 10^9$ documents would require over 1.2 TB of RAM, and sequential single-threaded streaming would bottleneck CPU throughput. Scaling to 2 TB requires:
1. Sharding the corpus into independent partitions and computing MinHash signatures in a distributed Ray / PySpark pipeline.
2. Replacing in-memory hash dictionaries with an external NVMe-backed RocksDB or distributed key-value store for exact deduplication.
3. Performing global near-deduplication via a two-stage MapReduce job: sorting by LSH band bucket hashes, grouping candidates in reducers, and performing localized shingle comparisons.
4. Reading directly from object storage via streaming Parquet row-group Arrow Flight readers without local disk staging.

---

### 8. Wrap-up

* **What we didn't get to (in priority order):**
  1. *Sub-dialect Register Tagging:* Granular classification separating formal Sanskritized Hindi (*Shuddh Hindi*), colloquial Urdu-influenced Hindustani, and regional code-switches (Bhojpuri/Maithili-mixed).
  2. *Tokenizer Perplexity Filtering:* Using a small Indic SentencePiece model to compute byte-level compression ratio and reject low-entropy machine translation noise.
  3. *Distributed Sharded Deduplication:* Implementing multi-process MinHash sharding across all CPU cores.
* **Actual Time Spent:**
  * Source Discovery & Downloading: 35 minutes
  * Pipeline Engineering (Streaming, Unicode Normalizer, LangID, Quality, PII): 1 hour 45 minutes
  * Deduplication & Contamination Detection: 45 minutes
  * Unit Testing, Verification, & DECISIONS.md Write-up: 35 minutes
  * **Total Time:** **3 hours 40 minutes** (within the 3–4 hour target).
