"""Checkpoint 1: deterministic, domain-independent column normalization."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import unicodedata

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


def normalize_column(name: str) -> str:
    """Normalize formatting while retaining words, numbers, and unit symbols.

    No spelling correction, abbreviation expansion, or semantic mapping occurs.
    Empty or separator-only input produces an empty string.
    """
    if not isinstance(name, str):
        raise TypeError("column name must be a string")
    text = unicodedata.normalize("NFC", name)
    # Unicode dash punctuation also includes en/em dashes and nonbreaking hyphens.
    text = "".join(
        " " if char == "_" or unicodedata.category(char) == "Pd" else char
        for char in text
    )
    # HTTPStatus -> HTTP Status; inputVoltage -> input Voltage.
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(text.lower().split())


def main() -> None:
    # Resolve .env relative to this file, even when invoked from another folder.
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Column names to normalize")
    parser.add_argument("--csv", type=Path, help="Read names from a CSV")
    parser.add_argument("--column", help="CSV field containing names; otherwise use headers")
    args = parser.parse_args()
    if args.names and args.csv:
        parser.error("use positional names or --csv, not both")
    if args.column is not None and args.csv is None:
        parser.error("--column requires --csv")
    if not args.names and args.csv is None:
        parser.error("provide names or --csv")

    names = args.names
    if args.csv is not None:
        try:
            with args.csv.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                headers = next(reader, [])
                if args.column is None:
                    names = headers
                else:
                    if headers.count(args.column) != 1:
                        parser.error("--column must identify exactly one CSV header")
                    index = headers.index(args.column)
                    names = []
                    for row in reader:
                        if not row:
                            continue
                        if index >= len(row):
                            parser.error(f"missing field at CSV line {reader.line_num}")
                        names.append(row[index])
        except (OSError, UnicodeError, csv.Error) as exc:
            parser.error(str(exc))

    # A list preserves duplicate inputs and makes formatting collisions visible.
    json.dump(
        [{"original": name, "normalized": normalize_column(name)} for name in names],
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
