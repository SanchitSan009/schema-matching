"""Run synthetic cluster consistency fixtures without API calls."""

import json
from calibrate import HERE
from cluster_confirmed import cluster_confirmed, validate_clusters


def fixture_report(case):
    pairs = []
    for key, band in [("matches", "HIGH_CONFIDENCE_MATCH"), ("non_matches", "HIGH_CONFIDENCE_NON_MATCH"), ("uncertain", "UNCERTAIN")]:
        for i, j in case[key]:
            pairs.append(dict(left_id=i, right_id=j, confidence_band=band))
    return dict(columns=[dict(id=i, original=chr(65 + i)) for i in range(case["size"])], pairs=pairs)


def main():
    benchmark = json.loads((HERE / "clustering_benchmark.json").read_text())
    rows = []
    for case in benchmark["cases"]:
        report = fixture_report(case)
        generated = cluster_confirmed(report)
        proposed = validate_clusters(report, [dict(member_ids=list(range(case["size"])))])
        passed = (sorted(len(c["member_ids"]) for c in generated["clusters"]) == case["expected_group_sizes"]
                  and proposed["passed"] == case["proposed_cluster_passes"] and generated["consistency_passed"])
        rows.append(dict(name=case["name"], passed=passed, generated=generated, proposed_cluster_audit=proposed))
    result = dict(synthetic=True, passed=all(r["passed"] for r in rows), cases=rows)
    output = HERE / "clustering_benchmark_report.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} synthetic fixtures passed. Full results: {output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
