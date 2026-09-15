"""Optional schema metadata and bounded, aggregate value evidence."""

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math

import numpy as np

FIELDS = ("table", "file", "data_type", "units", "description")


def metadata(record):
    result = {}
    for key in FIELDS:
        if key in record:
            if not isinstance(record[key], str):
                raise ValueError(f"{key} must be a string")
            result[key] = record[key][:2000]
    siblings = record.get("siblings", [])
    if not isinstance(siblings, list) or any(not isinstance(s, str) for s in siblings):
        raise ValueError("siblings must be a list of names")
    if siblings:
        result["siblings"] = [s[:200] for s in siblings[:20]]
        result["siblings_omitted"] = max(0, len(siblings) - 20)
    return result


def profile_values(values, limit=1000):
    if not isinstance(values, list) or type(limit) is not int or limit < 1:
        raise ValueError("values must be a list and limit positive")
    sample = values[:limit]
    nonnull = [v for v in sample if v is not None and v != ""]
    if any(isinstance(v, (list, dict)) for v in nonnull):
        raise ValueError("only scalar column values are supported")
    finite, formats, tokens = [], Counter(), set()
    for value in nonnull:
        text = str(value)
        tokens.add(hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        try:
            if isinstance(value, bool):
                raise ValueError()
            number = float(value)
            if not math.isfinite(number):
                raise ValueError()
            finite.append(number)
            formats["numeric"] += 1
        except (ValueError, TypeError):
            try:
                if not isinstance(value, str):
                    raise ValueError()
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                formats["iso_datetime"] += 1
            except ValueError:
                formats["boolean" if isinstance(value, bool) else "text"] += 1
    kind = "empty" if not nonnull else next(iter(formats)) if len(formats) == 1 else "mixed"
    result = dict(sample_count=len(sample), nonnull_count=len(nonnull),
                  observed_type=kind, missing_fraction=1 - len(nonnull) / len(sample) if sample else None,
                  distinct_count=len(tokens), formats=dict(formats),
                  sampling="first bounded values; not assumed representative", truncated=len(values) > limit,
                  category_hashes=sorted(tokens))
    if kind == "numeric":
        arr = np.asarray(finite, dtype=float)
        result["numeric"] = dict(min=float(arr.min()), max=float(arr.max()),
                                quantiles=np.quantile(arr, [0, .25, .5, .75, 1], method="nearest").tolist())
    return result


def public_profile(profile):
    return {k: v for k, v in profile.items() if k != "category_hashes"}


def value_evidence(left, right):
    a, b = left.get("value_profile"), right.get("value_profile")
    ma, mb = left.get("schema", {}), right.get("schema", {})
    units = "unknown" if not ma.get("units") or not mb.get("units") else "same_spelling" if ma["units"] == mb["units"] else "different_spelling_requires_interpretation"
    result = dict(available=bool(a and b), units=units, requires_review=units.startswith("different"))
    if not a or not b:
        return result
    result.update(left=public_profile(a), right=public_profile(b))
    kinds = {a["observed_type"], b["observed_type"]}
    result["type_agreement"] = a["observed_type"] == b["observed_type"]
    if len(kinds) > 1 and "empty" not in kinds:
        result["requires_review"] = True
    if a["observed_type"] == b["observed_type"] == "numeric":
        x, y = a["numeric"], b["numeric"]
        magnitude = max(abs(x["min"]), abs(x["max"]), abs(y["min"]), abs(y["max"]), 1.0)
        scale = max(x["max"] / magnitude, y["max"] / magnitude) - min(x["min"] / magnitude, y["min"] / magnitude)
        result["ranges_overlap"] = max(x["min"], y["min"]) <= min(x["max"], y["max"])
        result["normalized_quantile_distance"] = float(np.mean(np.abs(np.array(x["quantiles"]) / magnitude - np.array(y["quantiles"]) / magnitude))) / scale if scale else 0.0
    else:
        x, y = set(a.get("category_hashes", [])), set(b.get("category_hashes", []))
        result["categorical_jaccard"] = len(x & y) / len(x | y) if x | y else None
    result["interpretation"] = "Supporting sampled evidence only; overlap does not prove equivalence, disjoint ranges do not prove incompatibility. No unit conversion performed."
    return result


def enrich_records(records):
    groups = {}
    for r in records:
        groups.setdefault((r.get("table", ""), r.get("file", "")), []).append(r["name"])
    result = []
    for r in records:
        peers = groups[r.get("table", ""), r.get("file", "")]
        # Bound sibling context without generating an all-columns list per record.
        item = dict(r)
        item.setdefault("siblings", [n for n in peers[:21] if n != r["name"]][:20])
        result.append(item)
    return result


def csv_records(path, profile=False, row_limit=1000, cell_limit=1000000):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        names = next(reader, [])
        records = [dict(name=n, file=path.name) for n in names]
        if profile and names:
            limit = min(row_limit, cell_limit // len(names))
            if limit < 1:
                raise ValueError("CSV is too wide for the configured value sample budget")
            samples = [[] for _ in names]
            truncated = False
            for idx, row in enumerate(reader):
                if idx >= limit:
                    truncated = True
                    break
                if len(row) != len(names):
                    raise ValueError(f"CSV width mismatch on line {reader.line_num}")
                for sample, value in zip(samples, row):
                    sample.append(value)
            for record, sample in zip(records, samples):
                record["values"] = sample
                record["values_truncated"] = truncated
    return enrich_records(records)
