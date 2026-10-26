#!/usr/bin/env python3
"""
download.py - Download raw Indic corpora from verified public sources.
Records source metadata, download dates, and sha256 checksums.
"""

import os
import sys
import json
import urllib.request
import hashlib
from datetime import datetime, timezone
from pathlib import Path

SOURCES = [
    {
        "name": "aaditya/databricks-dolly-15k-Hindi",
        "url": "https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hindi/resolve/main/data/train-00000-of-00001.parquet",
        "filename": "dolly_15k_hindi.parquet",
        "license": "CC BY-SA 3.0",
        "description": "15,010 instruction-response documents translated to Hindi (Devanagari script)",
        "language_focus": "hi"
    },
    {
        "name": "DSMJ910/hinglish-instruct-10k",
        "url": "https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k/resolve/main/data/train-00000-of-00001.parquet",
        "filename": "hinglish_instruct_10k.parquet",
        "license": "Apache-2.0",
        "description": "10,594 Roman script Hinglish (code-mixed Hindi-English) conversational & instruction documents",
        "language_focus": "hi-Latn"
    },
    {
        "name": "l3cube-pune/code-mixed-nlp (L3Cube-HingLID)",
        "url": "https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/train.txt",
        "filename": "l3cube_hinglid_train.txt",
        "license": "CC BY-NC-SA 4.0",
        "description": "31,756 real-world Roman Hinglish social media utterances in token CoNLL format",
        "language_focus": "hi-Latn"
    }
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def download_file(url: str, dest: Path) -> None:
    print(f"Downloading: {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "IndicCorpusPipeline/1.0 (Mozilla/5.0 compatible)"}
    )
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 512 * 1024
        
        while chunk := response.read(chunk_size):
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / (1024*1024):.1f} MB downloaded", end="", flush=True)
        print()

def main():
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_records = []
    
    for src in SOURCES:
        dest_path = raw_dir / src["filename"]
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"Already exists: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB), skipping download.")
        else:
            download_file(src["url"], dest_path)
            
        sha256 = compute_sha256(dest_path)
        size_bytes = dest_path.stat().st_size
        
        record = {
            "name": src["name"],
            "url": src["url"],
            "license": src["license"],
            "description": src["description"],
            "language_focus": src["language_focus"],
            "download_date": datetime.now(timezone.utc).isoformat(),
            "local_file": src["filename"],
            "size_bytes": size_bytes,
            "sha256": sha256
        }
        metadata_records.append(record)
        print(f"Verified {src['filename']}: {size_bytes / (1024*1024):.2f} MB, sha256={sha256[:12]}...")

    sources_meta_path = raw_dir / "sources_metadata.json"
    with open(sources_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata recorded to {sources_meta_path}")

if __name__ == "__main__":
    main()
