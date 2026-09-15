"""Checkpoints 4–5: retrieve similar bases and generate column candidates."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from decompose import Parser, validate_response
from embeddings import BaseEmbedder, unit_vectors
from normalize import normalize_column

HERE = Path(__file__).resolve().parent


def retrieve_bases(bases, vectors, threshold=0.90, top_k=5, block_size=256):
    """Exact blocked cosine top-k, union of either direction, excluding self.

    Memory for similarity blocks is O(block_size * unique_bases), not O(columns²).
    Search computation is still quadratic in unique bases, not an ANN index.
    """
    if not -1 <= threshold <= 1 or top_k < 1 or block_size < 1:
        raise ValueError("threshold must be in [-1, 1]; top_k/block_size must be positive")
    if len(set(bases)) != len(bases):
        raise ValueError("retrieve_bases requires unique bases")
    if not bases:
        return []
    vectors = unit_vectors(vectors, len(bases))
    pairs = {}
    for start in range(0, len(bases), block_size):
        scores = np.clip(vectors[start:start + block_size] @ vectors.T, -1, 1)
        for offset, row in enumerate(scores):
            i = start + offset
            row[i] = -np.inf
            eligible = np.flatnonzero(row >= threshold)
            # Stable tie break by base index. Only sort threshold-passing neighbors.
            nearest = eligible[np.lexsort((eligible, -row[eligible]))[:top_k]]
            for j in nearest:
                pair = tuple(sorted((i, int(j))))
                pairs[pair] = float(row[j])
    return [dict(left_base=bases[i], right_base=bases[j], similarity=score)
            for (i, j), score in sorted(pairs.items())]


def checked_items(items):
    if not isinstance(items, list):
        raise ValueError("decompositions must be a list")
    result = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("original"), str):
            raise ValueError("each decomposition needs its original name")
        normalized = normalize_column(item["original"])
        parsed = validate_response({"items": [dict(item, id=0)]}, [normalized])[0]
        enriched = dict(original=item["original"], normalized=normalized, **parsed)
        for key in ("schema", "value_profile"):
            if key in item:
                enriched[key] = item[key]
        result.append(enriched)
    return result


def generate_candidates(items, embedder, threshold=0.90, top_k=15, max_pairs=100000,
                        representation="base", search_method="auto"):
    from retrieval import neighbors
    if not -1 <= threshold <= 1 or top_k < 1 or max_pairs < 1 or representation not in {"base", "full_name"}:
        raise ValueError("invalid retrieval settings")
    items = checked_items(items)
    bases = sorted({item["base"] for item in items})
    vectors = embedder.encode(bases) if len(items) > 1 else []
    base_vectors = dict(zip(bases, vectors))
    texts = [item["base"] if representation == "base" else item["normalized"] for item in items]
    if len(items) < 2:
        links, retrieval = [], dict(method="none", max_final_degree=0)
    else:
        if representation == "base":
            search_vectors = np.array([base_vectors[t] for t in texts])
        else:
            unique = list(dict.fromkeys(texts))
            lookup = dict(zip(unique, embedder.encode(unique)))
            search_vectors = np.array([lookup[t] for t in texts])
        links, retrieval = neighbors(search_vectors, top_k, threshold, search_method)
    if len(links) > max_pairs:
        raise ValueError(f"{len(links)} candidates exceed --max-pairs={max_pairs}")
    pairs = []
    for i, j, retrieval_score in links:
        left, right = items[i], items[j]
        exact = left["base"] == right["base"]
        base_score = 1.0 if exact else float(np.clip(base_vectors[left["base"]] @ base_vectors[right["base"]], -1, 1))
        if base_score < threshold:
            continue
        pairs.append(dict(left_id=i, right_id=j, left=left["original"], right=right["original"],
            base_similarity=base_score, retrieval_similarity=retrieval_score,
            retrieval_reason="exact_base" if exact else "semantic_base",
            decomposition_ambiguous=left["ambiguous"] or right["ambiguous"]))
    total = len(items) * (len(items) - 1) // 2
    return dict(stage="candidate_retrieval_only", timestamp=datetime.now(timezone.utc).isoformat(),
        embedding=embedder.settings, threshold=threshold, top_k=top_k,
        retrieval_version="bounded-column-v1", representation=representation, search=retrieval,
        column_count=len(items), unique_base_count=len(bases), all_possible_pairs=total,
        candidate_count=len(pairs), pair_reduction=1 - len(pairs) / total if total else 0.0,
        embedding_cache_hits=embedder.cache_hits, bases_embedded=embedder.embedded_count,
        columns=[dict(id=i, **item) for i, item in enumerate(items)],
        base_neighbors=[], candidates=pairs)


def read_csv_names(path, column=None):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
        if column is None:
            return headers
        if headers.count(column) != 1:
            raise ValueError("--column must identify exactly one CSV header")
        idx = headers.index(column)
        names = []
        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                raise ValueError(f"missing field at CSV line {reader.line_num}")
            names.append(row[idx])
        return names


def add_retrieval_options(cli):
    cli.add_argument("--backend", choices=["gemini", "sentence-transformers"])
    cli.add_argument("--embedding-model")
    cli.add_argument("--threshold", type=float, default=0.90,
                     help="Cosine cutoff; 0.90 is a provisional Gemini setting, tune per model")
    cli.add_argument("--top-k", type=int, default=15)
    cli.add_argument("--max-pairs", type=int, default=100000)
    cli.add_argument("--cache", type=Path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("names", nargs="*")
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path)
    source.add_argument("--decompositions", type=Path,
                        help="JSON from decompose.py or evaluate.py; reuse model output")
    cli.add_argument("--column", help="CSV field holding names; default: CSV headers")
    cli.add_argument("--parser-model")
    cli.add_argument("--output", type=Path, default=HERE / "candidate_report.json")
    add_retrieval_options(cli)
    args = cli.parse_args()
    if args.names and (args.csv or args.decompositions):
        cli.error("provide names, --csv, or --decompositions exclusively")
    if args.column is not None and args.csv is None:
        cli.error("--column requires --csv")
    if not -1 <= args.threshold <= 1 or args.top_k < 1 or args.max_pairs < 1:
        cli.error("invalid retrieval settings")
    if args.decompositions:
        payload = json.loads(args.decompositions.read_text())
        if isinstance(payload, list):
            items = payload
        elif "columns" in payload:
            items = payload["columns"]
        elif "rows" in payload:
            items = [r["actual"] for r in payload["rows"]]
        else:
            items = payload["items"]
    else:
        names = read_csv_names(args.csv, args.column) if args.csv else args.names
        if not names:
            cli.error("provide column names or a nonempty input file")
        items = Parser(args.parser_model).extract(names)
    embedder = BaseEmbedder(args.backend, args.embedding_model, args.cache)
    report = generate_candidates(items, embedder, args.threshold, args.top_k, args.max_pairs)
    report["decomposition_source"] = str(args.decompositions) if args.decompositions else "live_parser"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['candidate_count']} candidates / {report['all_possible_pairs']} possible pairs "
          f"({report['pair_reduction']:.1%} reduction)")
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
