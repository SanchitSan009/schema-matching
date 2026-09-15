"""Checkpoints 12–13: complete-link grouping and exhaustive consistency audit."""

import argparse
from itertools import combinations
import json
from pathlib import Path

from dotenv import load_dotenv
from normalize import ROOT
from calibrate import HERE, BANDS, confidence_report

MATCH = "HIGH_CONFIDENCE_MATCH"


def pair_index(report):
    ids = [c.get("id") for c in report["columns"]]
    if any(type(i) is not int for i in ids) or ids != list(range(len(ids))):
        raise ValueError("columns need consecutive integer IDs")
    indexed = {}
    for p in report["pairs"]:
        i, j = p["left_id"], p["right_id"]
        if type(i) is not int or type(j) is not int or not 0 <= i < j < len(ids) or (i, j) in indexed or p["confidence_band"] not in BANDS:
            raise ValueError("invalid confidence pair")
        indexed[i, j] = p
    return indexed


def evidence(indexed, i, j):
    pair = indexed.get(tuple(sorted((i, j))))
    status = "MISSING" if pair is None else (
        "EQUIVALENT" if pair["confidence_band"] == MATCH else
        "CONFLICTING" if pair["confidence_band"] == "HIGH_CONFIDENCE_NON_MATCH" else "UNCERTAIN")
    return dict(left_id=min(i, j), right_id=max(i, j), status=status,
                decision=None if pair is None else pair.get("decision"),
                relation=None if pair is None else pair.get("relation"))


def validate_clusters(report, clusters):
    indexed = pair_index(report)
    n = len(report["columns"])
    audited, assigned = [], set()
    for group in clusters:
        ids = group["member_ids"]
        if not ids or any(type(i) is not int or not 0 <= i < n for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("cluster members must be nonempty, valid unique IDs")
        if assigned.intersection(ids):
            raise ValueError("clusters must be disjoint")
        assigned.update(ids)
        pairs = [evidence(indexed, i, j) for i, j in combinations(sorted(ids), 2)]
        problems = [p for p in pairs if p["status"] != "EQUIVALENT"]
        witnesses = []
        for p in problems:
            a, c = p["left_id"], p["right_id"]
            # One witness per problematic closing edge suffices to show chaining.
            for b in ids:
                if b not in (a, c) and all(indexed.get(tuple(sorted(edge)), {}).get("confidence_band") == MATCH for edge in [(a, b), (b, c)]):
                    witnesses.append(dict(path=[a, b, c], closing_pair=p))
                    break
        audited.append(dict(member_ids=ids, members=[report["columns"][i]["original"] for i in ids],
                            status="FLAGGED" if problems else "SINGLETON" if len(ids) == 1 else "CONSISTENT",
                            checked_pair_count=len(pairs), confirmed_pair_count=len(pairs) - len(problems),
                            pair_evidence=pairs, problems=problems, chaining_witnesses=witnesses))
    return dict(passed=all(c["status"] != "FLAGGED" for c in audited), clusters=audited,
                unassigned_ids=sorted(set(range(n)) - assigned))


def cluster_confirmed(report):
    indexed = pair_index(report)
    n = len(report["columns"])
    groups = {i: {i} for i in range(n)}
    owner = list(range(n))
    # Deterministic, no raw-score preference. Complete cross-group evidence is required.
    edges = sorted(key for key, p in indexed.items() if p["confidence_band"] == MATCH)
    blocked = []
    for i, j in edges:
        a, b = owner[i], owner[j]
        if a == b:
            continue
        problem = next((evidence(indexed, x, y) for x in sorted(groups[a]) for y in sorted(groups[b])
                        if indexed.get(tuple(sorted((x, y))), {}).get("confidence_band") != MATCH), None)
        if problem:
            blocked.append(dict(trigger_pair=[i, j], left_group=sorted(groups[a]),
                                right_group=sorted(groups[b]), blocking_pair=problem))
            continue
        for member in groups[b]:
            owner[member] = a
        groups[a].update(groups.pop(b))
    clusters = [dict(member_ids=sorted(g)) for g in sorted(groups.values(), key=min)]
    audit = validate_clusters(report, clusters)
    return dict(stage="confirmed_clusters", clusters=audit["clusters"], consistency_passed=audit["passed"],
                blocked_merges=blocked, policy="all pairs must be HIGH_CONFIDENCE_MATCH; missing evidence blocks merge",
                column_count=n, cluster_count=len(clusters), non_singleton_count=sum(len(c["member_ids"]) > 1 for c in clusters))


def main():
    load_dotenv(ROOT / ".env", override=False)
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--decisions", type=Path, default=HERE / "decision_report.json")
    source.add_argument("--csv", type=Path, help="Run normalization through clustering on a CSV")
    cli.add_argument("--column", help="CSV field containing names; default: CSV headers")
    cli.add_argument("--context", default="")
    cli.add_argument("--profile", type=Path, default=HERE / "calibration_profile.json")
    cli.add_argument("--output", type=Path, default=HERE / "cluster_report.json")
    cli.add_argument("--validate-clusters", type=Path, help="Audit an external JSON list of {member_ids: [...]} groups")
    args = cli.parse_args()
    if args.column is not None and args.csv is None:
        cli.error("--column requires --csv")
    profile = json.loads(args.profile.read_text()) if args.profile.exists() else None
    if args.csv:
        from candidates import read_csv_names, generate_candidates
        from decompose import Parser
        from embeddings import BaseEmbedder
        from score import CONTEXT_FORMAT, score_candidates
        from relations import RelationClassifier
        from decide import decide_report
        names = read_csv_names(args.csv, args.column)
        if not names:
            cli.error("CSV must contain column names")
        parser, base_embedder = Parser(), BaseEmbedder()
        candidates = generate_candidates(parser.extract(names), base_embedder)
        candidates["parser_model"] = parser.model
        qualifiers = BaseEmbedder(base_embedder.backend, base_embedder.model, input_format=CONTEXT_FORMAT)
        decisions = decide_report(score_candidates(candidates, qualifiers), RelationClassifier(), context=args.context)
    else:
        decisions = json.loads(args.decisions.read_text())
    bands = confidence_report(decisions, profile)
    if args.validate_clusters:
        payload = json.loads(args.validate_clusters.read_text())
        result = validate_clusters(bands, payload if isinstance(payload, list) else payload["clusters"])
    else:
        result = cluster_confirmed(bands)
    result["confidence"] = bands
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"clusters", "confidence", "blocked_merges"}}))
    print(f"Full results: {args.output}")
    if args.validate_clusters and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
