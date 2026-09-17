"""Evaluate the canonical-variant relation policy on a focused fixture."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from relations import RelationClassifier

HERE = Path(__file__).resolve().parent


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--benchmark", type=Path, default=HERE / "semantic_policy_benchmark.json")
    cli.add_argument("--columns", type=Path,
                     default=HERE / "benchmark250_pipeline_report_semantic_bypass.json")
    cli.add_argument("--output", type=Path, default=HERE / "semantic_policy_report.json")
    args = cli.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    source = json.loads(args.columns.read_text())
    columns = {c["original"]: c for c in source["confidence"]["columns"]}
    requests = []
    for case in benchmark["cases"]:
        left, right = columns[case["left"]], columns[case["right"]]
        requests.append(dict(base_a=left["base"], qualifiers_a=left["qualifiers"],
                             base_b=right["base"], qualifiers_b=right["qualifiers"]))
    classifier = RelationClassifier()
    predictions = classifier.classify(requests)
    rows = [dict(case, actual_relation=prediction["relation"],
                 actual_base_relation=prediction["base_relation"],
                 correct=prediction["relation"] == case["expected_relation"],
                 prediction=prediction)
            for case, prediction in zip(benchmark["cases"], predictions)]
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  classifier=classifier.settings, count=len(rows),
                  correct=sum(row["correct"] for row in rows),
                  accuracy=sum(row["correct"] for row in rows) / len(rows),
                  predicted_relations=dict(Counter(row["actual_relation"] for row in rows)),
                  by_category={category: {
                      "count": sum(row["category"] == category for row in rows),
                      "correct": sum(row["category"] == category and row["correct"] for row in rows)}
                      for category in benchmark["categories"]},
                  rows=rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
