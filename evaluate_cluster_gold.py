"""Evaluate predicted clusters against an explicit gold partition."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_gold(path):
    groups = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label, values = line.split(":", 1)
        groups.append({"id": label.strip(), "members": [v.strip() for v in values.split(",")]})
    return groups


def pairs(groups, name_to_id):
    return {tuple(sorted((name_to_id[a], name_to_id[b])))
            for group in groups for a, b in combinations(group["members"], 2)}


def evaluate(report, gold):
    columns = report["confidence"]["columns"]
    names = [column["original"] for column in columns]
    name_to_id = {name: i for i, name in enumerate(names)}
    gold_names = [name for group in gold for name in group["members"]]
    if len(gold) != 85 or len(gold_names) != 250 or len(set(gold_names)) != 250:
        raise ValueError("gold must contain 85 disjoint clusters covering 250 names")
    if set(gold_names) != set(names):
        raise ValueError("gold names do not exactly match report columns")

    gold_pairs = pairs(gold, name_to_id)
    predicted_pairs = {tuple(sorted(pair)) for cluster in report["clusters"]
                       for pair in combinations(cluster["member_ids"], 2)}
    true_pairs = gold_pairs & predicted_pairs
    false_merges = predicted_pairs - gold_pairs
    missed_merges = gold_pairs - predicted_pairs
    precision = len(true_pairs) / len(predicted_pairs) if predicted_pairs else 0.0
    recall = len(true_pairs) / len(gold_pairs) if gold_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    decisions = {(p["left_id"], p["right_id"]): p for p in report["confidence"]["pairs"]}
    predicted_owner = {member: n for n, cluster in enumerate(report["clusters"])
                       for member in cluster["member_ids"]}
    missed_details = []
    for i, j in sorted(missed_merges):
        decision = decisions.get((i, j))
        if decision is None:
            cause = "retrieval_miss"
        elif decision["decision"] == "REJECT":
            cause = "gemini_reject"
        elif decision["decision"] == "UNCERTAIN":
            cause = "semantic_uncertain"
        else:
            cause = "complete_link_blocking"
        missed_details.append({
            "left_id": i, "right_id": j, "left": names[i], "right": names[j],
            "cause": cause, "decision": None if decision is None else decision["decision"],
            "decision_rule": None if decision is None else decision["decision_rule"],
            "relation": None if decision is None else decision.get("relation"),
            "left_predicted_cluster": predicted_owner[i],
            "right_predicted_cluster": predicted_owner[j],
        })
    false_details = [{"left_id": i, "right_id": j, "left": names[i], "right": names[j]}
                     for i, j in sorted(false_merges)]
    return {
        "gold_cluster_count": len(gold),
        "predicted_cluster_count": len(report["clusters"]),
        "gold_pair_count": len(gold_pairs),
        "predicted_pair_count": len(predicted_pairs),
        "true_positive_pairs": len(true_pairs),
        "false_merges": len(false_merges),
        "missed_merges": len(missed_merges),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "missed_merge_causes": dict(Counter(row["cause"] for row in missed_details)),
        "false_merge_pairs": false_details,
        "missed_merge_pairs": missed_details,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report", type=Path, default=HERE / "benchmark250_pipeline_report_semantic_bypass.json")
    cli.add_argument("--gold", type=Path, default=HERE / "benchmark250_gold_clusters.txt")
    cli.add_argument("--output", type=Path, default=HERE / "benchmark250_cluster_evaluation.json")
    args = cli.parse_args()
    result = evaluate(json.loads(args.report.read_text()), load_gold(args.gold))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"false_merge_pairs", "missed_merge_pairs"}}, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
