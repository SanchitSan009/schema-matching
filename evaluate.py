"""Run the manually labeled decomposition benchmark, without clustering."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from decompose import Parser, PROMPT, validate_response
from normalize import normalize_column

HERE = Path(__file__).resolve().parent


def evaluate(expected, predictions):
    if not expected or len(expected) != len(predictions):
        raise ValueError("benchmark must be nonempty and predictions must cover every case")
    rows = []
    for gold, actual in zip(expected, predictions):
        if actual["original"] != gold["name"]:
            raise ValueError("prediction order/name does not match benchmark")
        base_ok = actual["base"] == gold["base"]
        qualifiers_ok = Counter(actual["qualifiers"]) == Counter(gold["qualifiers"])
        rows.append(dict(name=gold["name"], expected={"base": gold["base"],
                         "qualifiers": gold["qualifiers"]}, actual=actual,
                         base_correct=base_ok, qualifiers_correct=qualifiers_ok,
                         exact_match=base_ok and qualifiers_ok))
    count = len(rows)
    return {"count": count,
            "base_accuracy": sum(r["base_correct"] for r in rows) / count,
            "qualifier_accuracy": sum(r["qualifiers_correct"] for r in rows) / count,
            "exact_accuracy": sum(r["exact_match"] for r in rows) / count,
            "ambiguous_count": sum(r["actual"]["ambiguous"] for r in rows),
            "rows": rows}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--benchmark", type=Path, default=HERE / "benchmark.json")
    cli.add_argument("--output", type=Path, default=HERE / "benchmark_report.json")
    cli.add_argument("--model")
    args = cli.parse_args()
    gold = json.loads(args.benchmark.read_text())
    if not isinstance(gold, list) or not gold:
        raise ValueError("benchmark must be a nonempty list")
    # Validate hand labels before spending an API call. Never send them to Parser.
    for row in gold:
        validate_response({"items": [dict(id=0, base=row["base"],
            qualifiers=row["qualifiers"], ambiguous=False, reason="label")]},
            [normalize_column(row["name"])])
    parser = Parser(args.model)
    predictions = parser.extract([r["name"] for r in gold])
    report = evaluate(gold, predictions)
    report.update(model=parser.model, timestamp=datetime.now(timezone.utc).isoformat(),
                  prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(args.benchmark.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
