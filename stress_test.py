import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from candidates import HERE, generate_candidates
from decompose import Parser
from embeddings import BaseEmbedder
from run import prepare_records
from score import CONTEXT_FORMAT, score_candidates
from relations import RelationClassifier
from decide import decide_report
from calibrate import confidence_report
from cluster_confirmed import cluster_confirmed
from error_taxonomy import categorize


def evaluate(benchmark, decisions, clustering, bands):
    predictions = {(p["left_id"], p["right_id"]): p for p in decisions["decisions"]}
    confidence = {(p["left_id"], p["right_id"]): p["confidence_band"] for p in bands["pairs"]}
    owner = {i: n for n, c in enumerate(clustering["clusters"]) for i in c["member_ids"]}
    rows = []
    for idx, case in enumerate(benchmark["cases"]):
        key = (2 * idx, 2 * idx + 1)
        prediction = predictions.get(key)
        actual = [decisions["columns"][i] for i in key]
        predicted_relation = prediction["relation"]["relation"] if prediction and prediction["relation"] else None
        expected_decision = "ACCEPT" if case["expected_relation"] == "EQUIVALENT" else "UNCERTAIN" if case["expected_relation"] == "UNCERTAIN" else "REJECT"
        tags = categorize(case, actual, prediction, prediction is not None, confidence.get(key), owner[key[0]] == owner[key[1]])
        rows.append(dict(id=case["id"], domain=case["domain"], expected_relation=case["expected_relation"],
                         predicted_relation=predicted_relation, retrieved=prediction is not None,
                         relation_correct=predicted_relation == case["expected_relation"],
                         decision_correct=prediction is not None and prediction["decision"] == expected_decision,
                         base_correct=all(a["base"] == e["base"] for a, e in zip(actual, case["columns"])),
                         errors=tags, prediction=prediction, actual_columns=actual))
    def summary(group):
        retained = [r for r in group if r["retrieved"]]
        positives = [r for r in group if r["expected_relation"] == "EQUIVALENT"]
        return dict(cases=len(group), base_pair_accuracy=sum(r["base_correct"] for r in group)/len(group),
                    retrieved=len(retained), positive_retrieval_recall=sum(r["retrieved"] for r in positives)/len(positives) if positives else None,
                    retained_relation_accuracy=sum(r["relation_correct"] for r in retained)/len(retained) if retained else None,
                    end_to_end_decision_accuracy=sum(r["decision_correct"] for r in group)/len(group),
                    error_counts=dict(Counter(tag for r in group for tag in r["errors"])))
    return dict(overall=summary(rows), domains={d:summary([r for r in rows if r["domain"]==d]) for d in sorted({r["domain"] for r in rows})}, rows=rows,
                attribution="Failure categories describe observed stages and benchmark phenomena, not proven causal diagnoses.")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--benchmark", type=Path, default=HERE / "cross_domain_benchmark.json")
    cli.add_argument("--output", type=Path, default=HERE / "cross_domain_report.json")
    cli.add_argument("--reuse-run", type=Path, help="Re-evaluate saved decisions without model calls")
    args = cli.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    if args.reuse_run:
        saved = json.loads(args.reuse_run.read_text())
        if saved["benchmark_sha256"] != hashlib.sha256(args.benchmark.read_bytes()).hexdigest():
            raise ValueError("saved run does not match this benchmark")
        decisions = saved["decisions"]
    else:
        parser, embedder = Parser(), BaseEmbedder()
        # Strip gold structure. Only names and optional schema facts reach the parser.
        records = [{k:v for k,v in c.items() if k not in {"base", "qualifiers"}} for case in benchmark["cases"] for c in case["columns"]]
        items = prepare_records(records, parser)
        # Each domain is an independent schema collection; no gold pair filtering.
        domains = {}
        for idx, case in enumerate(benchmark["cases"]):
            domains.setdefault(case["domain"], []).extend([2*idx, 2*idx+1])
        all_pairs = []
        retrieval_stats = {}
        for domain, ids in domains.items():
            part = generate_candidates([items[i] for i in ids], embedder, top_k=15, representation="full_name")
            retrieval_stats[domain] = {k:part[k] for k in ("candidate_count", "all_possible_pairs", "search")}
            for pair in part["candidates"]:
                all_pairs.append(dict(pair, left_id=ids[pair["left_id"]], right_id=ids[pair["right_id"]]))
        candidates = dict(columns=[dict(id=i, **item) for i,item in enumerate(items)], candidates=all_pairs,
                          embedding=embedder.settings, parser_model=parser.model, domain_retrieval=retrieval_stats,
                          retrieval_version="bounded-column-v1", representation="full_name", top_k=15, threshold=0.9)
        q = BaseEmbedder(embedder.backend, embedder.model, embedder.cache, input_format=CONTEXT_FORMAT)
        decisions = decide_report(score_candidates(candidates, q), RelationClassifier())
        decisions["evidence_version"] = "schema-values-v1"
    bands = confidence_report(decisions)
    clustering = cluster_confirmed(bands)
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), domain_holdout_status=benchmark["domain_holdout_status"],
                  benchmark_sha256=hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
                  evaluation=evaluate(benchmark, decisions, clustering, bands), decisions=decisions,
                  clusters=clustering, calibration="No held-out calibrated profile: bands abstain; singleton groups are not a semantic success claim.")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["evaluation"]["overall"],indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
