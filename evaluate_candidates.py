"""Measure retrieval recall/precision against an isolated manual smoke benchmark."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

from candidates import HERE, add_retrieval_options, generate_candidates
from decompose import Parser
from embeddings import BaseEmbedder


def retrieval_metrics(rows, candidates):
    expected = {(i, j) for i, j in combinations(range(len(rows)), 2)
                if rows[i]["candidate_group"] == rows[j]["candidate_group"]}
    actual = {(p["left_id"], p["right_id"]) for p in candidates}
    hits = len(actual & expected)
    return dict(expected_candidates=len(expected), true_positives=hits,
                false_positives=len(actual - expected), false_negatives=len(expected - actual),
                precision=hits / len(actual) if actual else None,
                recall=hits / len(expected) if expected else None,
                missed_pairs=[list(p) for p in sorted(expected - actual)],
                unexpected_pairs=[list(p) for p in sorted(actual - expected)])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--benchmark", type=Path, default=HERE / "candidate_benchmark.json")
    cli.add_argument("--output", type=Path, default=HERE / "candidate_benchmark_report.json")
    cli.add_argument("--gold-bases", action="store_true",
                     help="Isolate retrieval using annotated bases instead of calling the parser")
    cli.add_argument("--parser-model")
    add_retrieval_options(cli)
    args = cli.parse_args()
    if not -1 <= args.threshold <= 1 or args.top_k < 1 or args.max_pairs < 1:
        cli.error("invalid retrieval settings")
    rows = json.loads(args.benchmark.read_text())["columns"]
    if not rows:
        cli.error("benchmark is empty")
    if args.gold_bases:
        items = [dict(original=r["name"], base=r["base"], qualifiers=r["qualifiers"],
                      ambiguous=False, reason="Manual evaluation label") for r in rows]
        parser_model = None
    else:
        parser = Parser(args.parser_model)
        items = parser.extract([r["name"] for r in rows])
        parser_model = parser.model
    embedder = BaseEmbedder(args.backend, args.embedding_model, args.cache)
    report = generate_candidates(items, embedder, args.threshold, args.top_k, args.max_pairs)
    report["evaluation"] = retrieval_metrics(rows, report["candidates"])
    report["evaluation_mode"] = "gold_bases" if args.gold_bases else "live_parser"
    report["parser_model"] = parser_model
    report["benchmark_sha256"] = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["evaluation"], indent=2))
    print(f"Candidates: {report['candidate_count']}/{report['all_possible_pairs']}; "
          f"reduction: {report['pair_reduction']:.1%}")
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
