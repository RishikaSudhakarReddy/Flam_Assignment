"""
src/reader.py - Streaming raw record reader for Indic corpus pipeline.

Reads arbitrarily large datasets in a streaming fashion without loading
the entire corpus into memory.

Supports:
- Parquet files (.parquet)
- JSON Lines (.jsonl, .jsonl.gz)
- Plain text & CoNLL token files (.txt, .txt.gz)
- Tabular text files (.csv, .tsv)

Survives and reports:
- Malformed JSON / parsing errors
- Broken unicode / encoding errors
- Missing required fields
"""

import os
import json
import gzip
import csv
import io
from pathlib import Path
from typing import Generator, Dict, Any, Tuple, Optional


class StreamingReader:
    """
    Streaming reader that processes directory or individual files.
    Yields:
        (record_dict, is_valid, skip_reason)
        where record_dict is {'id': str, 'text': str, 'source': str, ...}
    """
    def __init__(self, raw_dir: str):
        self.raw_dir = Path(raw_dir)
        self.total_read = 0
        self.malformed_count = 0

    def iterate_records(self) -> Generator[Tuple[Optional[Dict[str, Any]], bool, Optional[str]], None, None]:
        """Iterate over all supported data files in raw_dir in a deterministic order."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {self.raw_dir}")

        files = sorted(list(self.raw_dir.glob("**/*")))
        
        for file_path in files:
            if file_path.is_dir() or file_path.name.startswith("."):
                continue
            if file_path.name in ("sources_metadata.json", "manifest.json"):
                continue

            source_name = file_path.name
            suffix = file_path.suffix.lower()

            if suffix == ".parquet":
                yield from self._read_parquet(file_path, source_name)
            elif suffix == ".gz" and file_path.stem.endswith(".jsonl"):
                yield from self._read_jsonl_gz(file_path, source_name)
            elif suffix == ".jsonl" or suffix == ".json":
                yield from self._read_jsonl(file_path, source_name)
            elif suffix == ".txt":
                yield from self._read_txt(file_path, source_name)
            elif suffix in (".csv", ".tsv"):
                delimiter = "\t" if suffix == ".tsv" else ","
                yield from self._read_delimited(file_path, source_name, delimiter)

    def _read_parquet(self, file_path: Path, source_name: str):
        """Stream parquet row-groups using pyarrow."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            # Fallback if pyarrow is unavailable
            yield None, False, f"pyarrow_missing_for_{file_path.name}"
            return

        try:
            parquet_file = pq.ParquetFile(str(file_path))
        except Exception as e:
            yield None, False, f"malformed_parquet_file: {str(e)}"
            return

        row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=1000):
            batch_dict = batch.to_pydict()
            num_rows = len(batch)
            cols = list(batch_dict.keys())

            for i in range(num_rows):
                row_idx += 1
                record_id = f"{source_name}#row_{row_idx}"
                
                try:
                    text = self._extract_text_from_dict(batch_dict, i, cols)
                    if text is None:
                        yield {"id": record_id, "source": source_name}, False, "no_valid_text_fields"
                    else:
                        yield {
                            "id": record_id,
                            "text": text,
                            "source": source_name
                        }, True, None
                except Exception as ex:
                    yield {"id": record_id, "source": source_name}, False, f"row_extraction_error: {str(ex)}"

    def _read_jsonl(self, file_path: Path, source_name: str):
        """Stream line-by-line JSONL file."""
        line_num = 0
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_num += 1
                record_id = f"{source_name}#L{line_num}"
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                except Exception as e:
                    yield {"id": record_id, "source": source_name}, False, f"malformed_json_syntax: {str(e)}"
                    continue

                text = self._extract_text_from_json(data)
                if text is None:
                    yield {"id": record_id, "source": source_name}, False, "missing_text_field"
                else:
                    rec_id = str(data.get("id", record_id))
                    yield {
                        "id": rec_id,
                        "text": text,
                        "source": source_name
                    }, True, None

    def _read_jsonl_gz(self, file_path: Path, source_name: str):
        """Stream line-by-line gzip-compressed JSONL."""
        line_num = 0
        try:
            with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_num += 1
                    record_id = f"{source_name}#L{line_num}"
                    line_str = line.strip()
                    if not line_str:
                        continue

                    try:
                        data = json.loads(line_str)
                    except Exception as e:
                        yield {"id": record_id, "source": source_name}, False, f"malformed_json_syntax: {str(e)}"
                        continue

                    text = self._extract_text_from_json(data)
                    if text is None:
                        yield {"id": record_id, "source": source_name}, False, "missing_text_field"
                    else:
                        rec_id = str(data.get("id", record_id))
                        yield {
                            "id": rec_id,
                            "text": text,
                            "source": source_name
                        }, True, None
        except Exception as e:
            yield {"id": f"{source_name}#archive", "source": source_name}, False, f"corrupted_gzip_archive: {str(e)}"

    def _read_txt(self, file_path: Path, source_name: str):
        """Stream text file; detects token CoNLL or standard document/paragraph structure."""
        line_num = 0
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Check if this is a CoNLL token dataset (like L3Cube-HingLID: token \t tag)
            first_line = f.readline()
            f.seek(0)
            is_conll = "\t" in first_line and len(first_line.split("\t")) == 2

            if is_conll:
                sentence_tokens = []
                sent_idx = 0
                for line in f:
                    line_num += 1
                    line_clean = line.strip()
                    if not line_clean:
                        if sentence_tokens:
                            sent_idx += 1
                            full_sent = " ".join(sentence_tokens)
                            sentence_tokens = []
                            yield {
                                "id": f"{source_name}#sent_{sent_idx}",
                                "text": full_sent,
                                "source": source_name
                            }, True, None
                    else:
                        parts = line_clean.split("\t")
                        if len(parts) >= 1 and parts[0].strip():
                            sentence_tokens.append(parts[0].strip())
                if sentence_tokens:
                    sent_idx += 1
                    yield {
                        "id": f"{source_name}#sent_{sent_idx}",
                        "text": " ".join(sentence_tokens),
                        "source": source_name
                    }, True, None
            else:
                # Standard paragraph / document text
                buffer = []
                doc_idx = 0
                for line in f:
                    line_num += 1
                    line_clean = line.strip()
                    if not line_clean:
                        if buffer:
                            doc_idx += 1
                            yield {
                                "id": f"{source_name}#doc_{doc_idx}",
                                "text": "\n".join(buffer),
                                "source": source_name
                            }, True, None
                            buffer = []
                    else:
                        buffer.append(line_clean)
                if buffer:
                    doc_idx += 1
                    yield {
                        "id": f"{source_name}#doc_{doc_idx}",
                        "text": "\n".join(buffer),
                        "source": source_name
                    }, True, None

    def _read_delimited(self, file_path: Path, source_name: str, delimiter: str):
        """Stream CSV/TSV row by row."""
        line_num = 0
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = None
            try:
                header = next(reader)
            except Exception as e:
                yield {"id": f"{source_name}#header", "source": source_name}, False, f"unreadable_csv_header: {str(e)}"
                return

            text_col_idx = None
            for idx, col in enumerate(header):
                if col.lower() in ("text", "content", "document", "sentence", "body"):
                    text_col_idx = idx
                    break

            for row in reader:
                line_num += 1
                record_id = f"{source_name}#row_{line_num}"
                if not row or not any(row):
                    continue
                try:
                    if text_col_idx is not None and text_col_idx < len(row):
                        text = row[text_col_idx]
                    else:
                        text = " ".join(cell.strip() for cell in row if cell.strip())
                    yield {
                        "id": record_id,
                        "text": text,
                        "source": source_name
                    }, True, None
                except Exception as e:
                    yield {"id": record_id, "source": source_name}, False, f"csv_parse_error: {str(e)}"

    def _extract_text_from_dict(self, batch_dict: dict, row_idx: int, cols: list) -> Optional[str]:
        """Extract text from pyarrow batch dict for standard known and generic schemas."""
        # 1. Dolly Hindi schema
        if "hindi_instruction" in cols and "hindi_output" in cols:
            inst = str(batch_dict["hindi_instruction"][row_idx] or "").strip()
            inp = str(batch_dict.get("hindi_input", [""] * (row_idx + 1))[row_idx] or "").strip()
            out = str(batch_dict["hindi_output"][row_idx] or "").strip()
            parts = [p for p in (inst, inp, out) if p]
            return "\n\n".join(parts) if parts else None

        # 2. Hinglish Instruct schema
        if "user_message" in cols and "assistant_response" in cols:
            user = str(batch_dict["user_message"][row_idx] or "").strip()
            resp = str(batch_dict["assistant_response"][row_idx] or "").strip()
            parts = [p for p in (user, resp) if p]
            return "\n\n".join(parts) if parts else None

        # 3. Standard text / content columns
        for key in ("text", "content", "document", "sentence", "body", "input", "output"):
            if key in cols:
                val = batch_dict[key][row_idx]
                if val:
                    return str(val)

        # 4. Fallback: join string columns
        string_vals = [
            str(batch_dict[c][row_idx])
            for c in cols
            if batch_dict[c][row_idx] is not None and isinstance(batch_dict[c][row_idx], str)
        ]
        return "\n\n".join(string_vals) if string_vals else None

    def _extract_text_from_json(self, data: Any) -> Optional[str]:
        """Extract text from JSON object."""
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return str(data)

        # Check standard text keys
        for key in ("text", "content", "document", "sentence", "body", "prompt", "output"):
            if key in data and data[key]:
                return str(data[key])

        # Instruct style
        if "user_message" in data or "assistant_response" in data:
            parts = [str(data.get(k, "")).strip() for k in ("user_message", "assistant_response") if data.get(k)]
            return "\n\n".join(parts) if parts else None

        # Generic: join all string values
        vals = [str(v).strip() for v in data.values() if isinstance(v, str) and v.strip()]
        return "\n\n".join(vals) if vals else None
