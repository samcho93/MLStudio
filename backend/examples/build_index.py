"""
build_index.py
Scans all example JSON files in subdirectories and generates _index.json.

Usage:
    python build_index.py
"""

import json
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SUBDIRS = [
    "tutorial",
    "classification",
    "regression",
    "unsupervised",
    "image",
    "nlp",
    "timeseries",
    "advanced",
]

INDEX_FIELDS = [
    "id",
    "title",
    "title_ko",
    "category",
    "difficulty",
    "tags",
    "estimated_time",
    "requires_phase",
]


def main():
    entries = []

    for subdir in SUBDIRS:
        dir_path = os.path.join(SCRIPT_DIR, subdir)
        if not os.path.isdir(dir_path):
            print(f"WARNING: directory not found: {dir_path}")
            continue

        for filepath in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"ERROR reading {filepath}: {e}")
                continue

            meta = data.get("meta")
            if meta is None:
                print(f"WARNING: no 'meta' section in {filepath}")
                continue

            entry = {}
            for field in INDEX_FIELDS:
                if field in meta:
                    entry[field] = meta[field]

            if "id" not in entry:
                print(f"WARNING: no 'id' in meta of {filepath}")
                continue

            entries.append(entry)

    # Sort by id
    entries.sort(key=lambda e: e["id"])

    # Check count
    print(f"Total examples indexed: {len(entries)}")

    ids = [e["id"] for e in entries]
    expected = list(range(1, 101))
    missing = set(expected) - set(ids)
    if missing:
        print(f"WARNING: missing IDs: {sorted(missing)}")

    # Write _index.json
    output_path = os.path.join(SCRIPT_DIR, "_index.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
