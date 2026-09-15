"""Checkpoint 9: report relation quality by class and difficulty stratum."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from relations import HERE, RELATIONS, BASE_RELATIONS, RelationClassifier, canonical_request


def relation_metrics(cases, predictions):
    if not cases or len(cases) != len(predictions):
        raise ValueError("predictions must cover every case in a nonempty benchmark")
    rows = [dict(case, actual=actual, correct=case["expected"] == actual["relation"],
                 base_correct=case["expected_base"] == actual["base_relation"])
            for case, actual in zip(cases, predictions)]

    def summary(group):
        return dict(count=len(group), correct=sum(r["correct"] for r in group),
                    accuracy=sum(r["correct"] for r in group) / len(group) if group else None)

    matrix = {expected: {actual: 0 for actual in RELATIONS} for expected in RELATIONS}
    for row in rows:
        matrix[row["expected"]][row["actual"]["relation"]] += 1
    return dict(overall=summary(rows),
                by_stratum={s: summary([r for r in rows if r["stratum"] == s]) for s in sorted({r["stratum"] for r in rows})},
                by_relation={s: summary([r for r in rows if r["expected"] == s]) for s in RELATIONS},
                base_accuracy=sum(r["base_correct"] for r in rows) / len(rows),
                hard_negatives_called_equivalent=sum(r["stratum"] == "hard_negatives" and r["actual"]["relation"] == "EQUIVALENT" for r in rows),
                uncertain_cases_forced_equivalent=sum(r["expected"] == "UNCERTAIN" and r["actual"]["relation"] == "EQUIVALENT" for r in rows),
                confusion_matrix=matrix, rows=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--benchmark", type=Path, default=HERE / "relation_benchmark.json")
    cli.add_argument("--output", type=Path, default=HERE / "relation_benchmark_report.json")
    cli.add_argument("--model")
    cli.add_argument("--reuse-cache", action="store_true", help="Default evaluation uses fresh LLM calls")
    args = cli.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    cases = benchmark["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        cli.error("benchmark must contain unique cases")
    for case in cases:
        canonical_request(case["input"])
        if case["expected"] not in RELATIONS or case["expected_base"] not in BASE_RELATIONS or case["stratum"] not in {"positives", "hard_negatives", "context_dependent"}:
            cli.error("invalid benchmark labels or stratum")
    classifier = RelationClassifier(args.model)
    predictions = classifier.classify([case["input"] for case in cases], fresh=not args.reuse_cache)
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), classifier=classifier.settings,
                  fresh=not args.reuse_cache, benchmark_sha256=hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
                  **relation_metrics(cases, predictions))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "by_stratum", "hard_negatives_called_equivalent", "uncertain_cases_forced_equivalent")}, indent=2))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
