"""Evaluate contextual qualifiers, isolated baseline, and weighted scores."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from candidates import HERE, add_retrieval_options, generate_candidates
from decompose import Parser
from embeddings import BaseEmbedder
from score import CONTEXT_FORMAT, add_score_options, score_candidates, validate_weights


def auc(rows, field):
    positives = [r[field] for r in rows if r["equivalent"]]
    negatives = [r[field] for r in rows if not r["equivalent"]]
    if not positives or not negatives:
        return None
    return sum(1 if p > n else 0.5 if p == n else 0 for p in positives for n in negatives) / (len(positives) * len(negatives))


def evaluate_scores(benchmark, scored, cutoff=0.9):
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("diagnostic cutoff must be in [0, 1]")
    indexed = {(p["left_id"], p["right_id"]): p for p in scored["scored_candidates"]}
    rows = []
    for label in benchmark["pairs"]:
        key = (label["left_id"], label["right_id"])
        actual = indexed.get(key)
        row = dict(label, left=benchmark["columns"][key[0]]["name"],
                   right=benchmark["columns"][key[1]]["name"], retrieved=actual is not None,
                   above_diagnostic_cutoff=actual is not None and actual["equivalence_score"] >= cutoff)
        if actual:
            row.update(equivalence_score=actual["equivalence_score"],
                       contextual=actual["qualifier_cosine"],
                       isolated=actual["isolated_qualifier_cosine"],
                       lexical=actual["signals"]["lexical"], base=actual["signals"]["base"])
        rows.append(row)
    retained = [r for r in rows if r["retrieved"]]
    tp = sum(r["equivalent"] and r["above_diagnostic_cutoff"] for r in rows)
    fp = sum(not r["equivalent"] and r["above_diagnostic_cutoff"] for r in rows)
    fn = sum(r["equivalent"] and not r["above_diagnostic_cutoff"] for r in rows)
    rankings = []
    for comparison in benchmark.get("rankings", []):
        positive, negative = indexed.get(tuple(comparison["positive"])), indexed.get(tuple(comparison["negative"]))
        row = dict(comparison, evaluated=positive is not None and negative is not None)
        if row["evaluated"]:
            for field in ("qualifier_cosine", "isolated_qualifier_cosine", "equivalence_score"):
                margin = positive[field] - negative[field]
                row[field] = dict(margin=margin, correct_order=margin > 0)
        rankings.append(row)
    positives = sum(r["equivalent"] for r in rows)
    return dict(labeled_pairs=len(rows), labeled_pairs_retrieved=len(retained),
                positive_retrieval_recall=sum(r["equivalent"] and r["retrieved"] for r in rows) / positives if positives else None,
                retained_pair_auc={f: auc(retained, f) for f in ("contextual", "isolated", "equivalence_score")},
                diagnostic_cutoff=cutoff, true_positives=tp, false_positives=fp, false_negatives=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                rows=rows, rankings=rankings)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--benchmark", type=Path, default=HERE / "equivalence_benchmark.json")
    cli.add_argument("--output", type=Path, default=HERE / "equivalence_benchmark_report.json")
    cli.add_argument("--live-parser", action="store_true", help="Use LLM decomposition instead of manual benchmark structure")
    cli.add_argument("--parser-model")
    cli.add_argument("--diagnostic-cutoff", type=float, default=0.9,
                     help="Evaluation only; not a production equivalence threshold")
    add_retrieval_options(cli)
    add_score_options(cli)
    args = cli.parse_args()
    try:
        validate_weights(args.weights)
    except ValueError as exc:
        cli.error(str(exc))
    if not 0 <= args.diagnostic_cutoff <= 1 or not -1 <= args.threshold <= 1 or args.top_k < 1 or args.max_pairs < 1:
        cli.error("invalid retrieval or diagnostic settings")
    benchmark = json.loads(args.benchmark.read_text())
    if not benchmark["columns"] or not benchmark["pairs"]:
        cli.error("benchmark needs columns and labeled pairs")
    seen = set()
    for pair in benchmark["pairs"]:
        i, j = pair["left_id"], pair["right_id"]
        if (type(i) is not int or type(j) is not int or not 0 <= i < j < len(benchmark["columns"])
                or type(pair["equivalent"]) is not bool or (i, j) in seen):
            cli.error("invalid or duplicate benchmark pair")
        seen.add((i, j))
    parser_model = None
    if args.live_parser:
        parser = Parser(args.parser_model)
        items = parser.extract([r["name"] for r in benchmark["columns"]])
        parser_model = parser.model
    else:
        items = [dict(original=r["name"], base=r["base"], qualifiers=r["qualifiers"],
                      ambiguous=False, reason="Manual evaluation decomposition") for r in benchmark["columns"]]
    base_embedder = BaseEmbedder(args.backend, args.embedding_model, args.cache)
    candidates = generate_candidates(items, base_embedder, args.threshold, args.top_k, args.max_pairs)
    qualifier_embedder = BaseEmbedder(base_embedder.backend, base_embedder.model, args.cache, input_format=CONTEXT_FORMAT)
    report = score_candidates(candidates, qualifier_embedder, args.weights, isolated_baseline=True)
    report["evaluation_mode"] = "live_parser" if args.live_parser else "manual_decomposition"
    report["parser_model"] = parser_model
    report["benchmark_sha256"] = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    report["evaluation"] = evaluate_scores(benchmark, report, args.diagnostic_cutoff)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report["evaluation"].items() if k not in {"rows", "rankings"}}, indent=2))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
