"""Checkpoint 11: empirical reliability bands with conservative abstention."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

from dotenv import load_dotenv
from normalize import ROOT

HERE = Path(__file__).resolve().parent
BANDS = ("HIGH_CONFIDENCE_MATCH", "HIGH_CONFIDENCE_NON_MATCH", "UNCERTAIN")
SCORE_EDGES = [0.0, 0.8, 0.9, 0.95, 1.00000001]


def signature(report):
    scoring = report.get("scoring_provenance", {})
    retrieval = scoring.get("retrieval", {})
    settings = dict(classifier=report.get("classifier"), policy=report.get("decision_policy"),
                    thresholds=report.get("support_thresholds"), context=report.get("context", ""),
                    embedding=scoring.get("qualifier_embedding"), weights=scoring.get("weights"),
                    parser_model=retrieval.get("parser_model"),
                    retrieval_settings={k: retrieval.get(k) for k in ("threshold", "top_k", "embedding", "retrieval_version", "representation")},
                    evidence_version=report.get("evidence_version"))
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def checked_pairs(report):
    ids = [c.get("id") for c in report["columns"]]
    if any(type(i) is not int for i in ids) or ids != list(range(len(ids))):
        raise ValueError("columns need consecutive integer IDs")
    seen = set()
    for pair in report["decisions"]:
        i, j = pair["left_id"], pair["right_id"]
        if type(i) is not int or type(j) is not int or not 0 <= i < j < len(ids) or (i, j) in seen:
            raise ValueError("invalid or duplicate pair IDs")
        if pair["decision"] not in {"ACCEPT", "REJECT", "UNCERTAIN"}:
            raise ValueError("invalid pair decision")
        seen.add((i, j))
        bucket(pair)
    return report["decisions"]


def bucket(pair):
    score = pair["equivalence_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("raw equivalence score must be finite and in [0, 1]")
    idx = next(i for i in range(len(SCORE_EDGES) - 1) if SCORE_EDGES[i] <= score < SCORE_EDGES[i + 1])
    return f"{pair['decision']}|{pair['decision_rule']}|score_bin_{idx}"


def wilson(successes, count, confidence=0.95):
    if count <= 0 or not 0 <= successes <= count or not 0 < confidence < 1:
        raise ValueError("invalid Wilson interval inputs")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def fit_profile(report, benchmark, min_samples=20, target_precision=0.9, confidence=0.95):
    if type(min_samples) is not int or min_samples < 1 or not 0 < target_precision <= 1 or not 0 < confidence < 1:
        raise ValueError("invalid calibration settings")
    pairs = checked_pairs(report)
    if [c["original"] for c in report["columns"]] != [c["name"] for c in benchmark["columns"]]:
        raise ValueError("calibration labels must align with report columns")
    indexed = {(p["left_id"], p["right_id"]): p for p in pairs}
    bins, seen = {}, set()
    misses, uncertain = 0, 0
    for label in benchmark["pairs"]:
        i, j = label["left_id"], label["right_id"]
        if type(i) is not int or type(j) is not int or not 0 <= i < j < len(report["columns"]) or (i, j) in seen or type(label["equivalent"]) is not bool:
            raise ValueError("invalid or duplicate calibration label")
        seen.add((i, j))
        pair = indexed.get((i, j))
        if pair is None:
            misses += 1
            continue
        if pair["decision"] == "UNCERTAIN":
            uncertain += 1
            continue
        entry = bins.setdefault(bucket(pair), dict(count=0, correct=0))
        entry["count"] += 1
        entry["correct"] += (pair["decision"] == "ACCEPT") == label["equivalent"]
    if not seen:
        raise ValueError("calibration labels must be nonempty")
    for entry in bins.values():
        entry["empirical_correctness"] = entry["correct"] / entry["count"]
        entry["correctness_interval"] = wilson(entry["correct"], entry["count"], confidence)
        entry["eligible_for_high_confidence"] = entry["count"] >= min_samples and entry["correctness_interval"][0] >= target_precision
    return dict(version="empirical-bands-v1", signature=signature(report), score_edges=SCORE_EDGES,
                created_at=datetime.now(timezone.utc).isoformat(), min_samples=min_samples,
                target_precision=target_precision, interval_confidence=confidence, bins=bins,
                labeled_count=len(seen), not_retrieved=misses, uncertain_decisions=uncertain,
                high_confidence_bin_count=sum(b["eligible_for_high_confidence"] for b in bins.values()),
                labels_sha256=hashlib.sha256(json.dumps(benchmark, sort_keys=True).encode()).hexdigest(),
                limitation="Empirical development-data reliability, not an independently validated individual probability. Correlated labels and domain shift weaken coverage.")


def confidence_report(report, profile=None):
    pairs = checked_pairs(report)
    if profile is not None and (profile.get("version") != "empirical-bands-v1" or profile.get("score_edges") != SCORE_EDGES):
        raise ValueError("unsupported calibration profile")
    compatible = profile is not None and profile.get("signature") == signature(report)
    results = []
    for pair in pairs:
        entry = profile["bins"].get(bucket(pair)) if compatible else None
        columns = [report["columns"][pair["left_id"]], report["columns"][pair["right_id"]]]
        ambiguous = any(c.get("ambiguous", True) for c in columns)
        mismatch = bool(columns[0]["qualifiers"]) != bool(columns[1]["qualifiers"])
        relation = pair.get("relation") or {}
        semantic_uncertainty = (relation.get("base_relation") == "UNCERTAIN" or
                                (relation.get("relation") == "UNCERTAIN" and relation.get("base_relation") != "INCOMPATIBLE"))
        band, reason = "UNCERTAIN", "insufficient_calibration_evidence"
        if pair["decision"] == "UNCERTAIN" or ambiguous or mismatch or semantic_uncertainty:
            reason = "semantic_or_scope_uncertainty"
        elif not compatible:
            reason = "missing_or_incompatible_calibration_profile"
        elif entry and entry["count"] >= profile["min_samples"] and wilson(entry["correct"], entry["count"], profile["interval_confidence"])[0] >= profile["target_precision"]:
            band = "HIGH_CONFIDENCE_MATCH" if pair["decision"] == "ACCEPT" else "HIGH_CONFIDENCE_NON_MATCH"
            reason = "empirical_reliability_lower_bound_passed"
        results.append(dict(pair, confidence_band=band, confidence_reason=reason, calibration_evidence=entry))
    return dict(stage="confidence_bands", columns=report["columns"], pairs=results,
                calibration_profile=profile, profile_compatible=compatible,
                counts=dict(Counter(p["confidence_band"] for p in results)),
                source_signature=signature(report))


def main():
    load_dotenv(ROOT / ".env", override=False)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=["fit", "apply"])
    cli.add_argument("--decisions", type=Path, default=HERE / "decision_benchmark_report.json")
    cli.add_argument("--labels", type=Path, default=HERE / "equivalence_benchmark.json")
    cli.add_argument("--profile", type=Path, default=HERE / "calibration_profile.json")
    cli.add_argument("--output", type=Path, default=HERE / "confidence_report.json")
    cli.add_argument("--min-samples", type=int, default=20)
    cli.add_argument("--target-precision", type=float, default=0.9)
    args = cli.parse_args()
    report = json.loads(args.decisions.read_text())
    if args.mode == "fit":
        result = fit_profile(report, json.loads(args.labels.read_text()), args.min_samples, args.target_precision)
        output = args.profile
        summary = {k: result[k] for k in ("labeled_count", "high_confidence_bin_count")}
    else:
        result = confidence_report(report, json.loads(args.profile.read_text()))
        output, summary = args.output, result["counts"]
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))
    print(f"Full results: {output}")


if __name__ == "__main__":
    main()
