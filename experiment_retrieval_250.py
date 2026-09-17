"""Controlled cached retrieval experiment for the 250-column gold partition."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from candidates import generate_candidates
from embeddings import BaseEmbedder
from evaluate_cluster_gold import load_gold
from retrieval import neighbors

HERE = Path(__file__).resolve().parent


def gold_pairs(groups, by_name):
    return {tuple(sorted((by_name[a], by_name[b])))
            for group in groups for a, b in combinations(group["members"], 2)}


def run(items, gold, threshold, top_k, embedder):
    part = generate_candidates(items, embedder, threshold=threshold, top_k=top_k,
                               representation="full_name", search_method="exact")
    if part["bases_embedded"]:
        raise RuntimeError("embedding cache miss; aborting controlled experiment")
    names = [item["original"] for item in items]
    by_name = {name: i for i, name in enumerate(names)}
    expected = gold_pairs(gold, by_name)
    retained = {(p["left_id"], p["right_id"]) for p in part["candidates"]}
    hits = expected & retained
    return part, expected, retained, hits


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path,
                     default=HERE / "benchmark250_pipeline_report_semantic_bypass.json")
    cli.add_argument("--gold", type=Path, default=HERE / "benchmark250_gold_clusters.txt")
    cli.add_argument("--output", type=Path,
                     default=HERE / "benchmark250_retrieval_threshold_experiment.json")
    args = cli.parse_args()
    source = json.loads(args.source.read_text())
    items = source["confidence"]["columns"]
    gold = load_gold(args.gold)
    settings = source["confidence"]["pairs"][0]
    # Both retrieval representations use the production bare-base embedding cache.
    embedder = BaseEmbedder("gemini", "gemini-embedding-001", input_format="bare-base-v1")
    baseline, expected, retained, baseline_hits = run(items, gold, 0.90, 15, embedder)
    test, _, _, test_hits = run(items, gold, 0.88, 15, embedder)
    top_k_test, _, _, top_k_hits = run(items, gold, 0.88, 25, embedder)

    names = [item["original"] for item in items]
    full_vectors = embedder.encode([item["normalized"] for item in items])
    unique_bases = list(dict.fromkeys(item["base"] for item in items))
    base_vectors = dict(zip(unique_bases, embedder.encode(unique_bases)))
    if embedder.embedded_count:
        raise RuntimeError("embedding cache miss during diagnosis; aborting")
    full_name_links, full_name_search = neighbors(full_vectors, k=15, threshold=0.88,
                                                  method="exact")
    full_name_only = {(i, j) for i, j, _ in full_name_links}
    full_name_hits = expected & full_name_only
    diagnoses = []
    for i, j in sorted(expected - retained):
        full_similarity = float(np.clip(full_vectors[i] @ full_vectors[j], -1, 1))
        base_similarity = (1.0 if items[i]["base"] == items[j]["base"] else
                           float(np.clip(base_vectors[items[i]["base"]] @
                                         base_vectors[items[j]["base"]], -1, 1)))
        if base_similarity < 0.90:
            category = "C_base_or_decomposition_too_different"
        elif full_similarity < 0.90:
            category = "A_similarity_below_threshold"
        else:
            category = "B_dropped_by_top_k_or_degree_cap"
        diagnoses.append(dict(left_id=i, right_id=j, left=names[i], right=names[j],
                              left_base=items[i]["base"], right_base=items[j]["base"],
                              full_name_similarity=full_similarity,
                              base_similarity=base_similarity, category=category,
                              recovered_at_088=(i, j) in test_hits))

    def summary(part, hits):
        return dict(threshold=part["threshold"], top_k=part["top_k"],
                    gold_pair_count=len(expected), gold_pair_hits=len(hits),
                    gold_pair_retrieval_recall=len(hits) / len(expected),
                    candidate_count=part["candidate_count"],
                    degree_drops=part["search"]["degree_drops"],
                    embedding_cache_hits=part["embedding_cache_hits"],
                    bases_embedded=part["bases_embedded"])

    result = dict(
        baseline=summary(baseline, baseline_hits),
        test=summary(test, test_hits),
        top_k_test=summary(top_k_test, top_k_hits),
        full_name_only_test=dict(
            threshold=0.88, top_k=15, secondary_base_gate=False,
            gold_pair_count=len(expected), gold_pair_hits=len(full_name_hits),
            gold_pair_retrieval_recall=len(full_name_hits) / len(expected),
            candidate_count=len(full_name_only),
            degree_drops=full_name_search["degree_drops"],
            embedding_cache_hits=embedder.cache_hits, bases_embedded=embedder.embedded_count),
        recall_gain=(len(test_hits) - len(baseline_hits)) / len(expected),
        additional_gold_pairs=len(test_hits - baseline_hits),
        additional_candidates=test["candidate_count"] - baseline["candidate_count"],
        candidate_growth=(test["candidate_count"] / baseline["candidate_count"] - 1),
        top_k_additional_gold_pairs=len(top_k_hits - test_hits),
        top_k_additional_candidates=top_k_test["candidate_count"] - test["candidate_count"],
        top_k_candidate_growth=(top_k_test["candidate_count"] / test["candidate_count"] - 1),
        no_base_gate_additional_gold_pairs=len(full_name_hits - test_hits),
        no_base_gate_additional_candidates=len(full_name_only) - test["candidate_count"],
        no_base_gate_candidate_growth=len(full_name_only) / test["candidate_count"] - 1,
        baseline_miss_categories=dict(Counter(d["category"] for d in diagnoses)),
        baseline_miss_diagnoses=diagnoses,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "baseline_miss_diagnoses"}, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
