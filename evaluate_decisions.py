"""Evaluate relation-led decisions on the earlier scored equivalence benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

from candidates import HERE
from decide import decide_report
from relations import RelationClassifier


def decision_metrics(benchmark, report):
    columns = report["columns"]
    if [c["original"] for c in columns] != [c["name"] for c in benchmark["columns"]]:
        raise ValueError("decision report and benchmark columns do not align")
    decisions = {(r["left_id"], r["right_id"]): r for r in report["decisions"]}
    rows = []
    for pair in benchmark["pairs"]:
        result = decisions.get((pair["left_id"], pair["right_id"]))
        rows.append(dict(pair, decision=result["decision"] if result else "NOT_RETRIEVED"))
    tp = sum(r["equivalent"] and r["decision"] == "ACCEPT" for r in rows)
    fp = sum(not r["equivalent"] and r["decision"] == "ACCEPT" for r in rows)
    determinate = [r for r in rows if r["decision"] in {"ACCEPT", "REJECT"}]
    positives = sum(r["equivalent"] for r in rows)
    return dict(labeled_pairs=len(rows), accepted_true_positives=tp, false_accepts=fp,
                accept_precision=tp / (tp + fp) if tp + fp else None,
                accept_recall=tp / positives if positives else None,
                uncertain=sum(r["decision"] == "UNCERTAIN" for r in rows),
                not_retrieved=sum(r["decision"] == "NOT_RETRIEVED" for r in rows),
                decision_coverage=len(determinate) / len(rows) if rows else None,
                determinate_accuracy=sum((r["decision"] == "ACCEPT") == r["equivalent"] for r in determinate) / len(determinate) if determinate else None,
                rows=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scores", type=Path, default=HERE / "equivalence_benchmark_report.json")
    cli.add_argument("--benchmark", type=Path, default=HERE / "equivalence_benchmark.json")
    cli.add_argument("--output", type=Path, default=HERE / "decision_benchmark_report.json")
    cli.add_argument("--model")
    cli.add_argument("--fresh", action="store_true")
    args = cli.parse_args()
    scored = json.loads(args.scores.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    if [c["original"] for c in scored["columns"]] != [c["name"] for c in benchmark["columns"]]:
        cli.error("scored columns must align with the benchmark")
    report = decide_report(scored, RelationClassifier(args.model), fresh=args.fresh)
    report["evaluation"] = decision_metrics(benchmark, report)
    report["benchmark_sha256"] = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report["evaluation"].items() if k != "rows"}, indent=2))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
