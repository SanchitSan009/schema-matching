"""Re-run cross-domain retrieval with saved parser output and cached embeddings."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from candidates import HERE, generate_candidates
from embeddings import BaseEmbedder


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--threshold", type=float, required=True)
    cli.add_argument("--top-k", type=int, required=True)
    cli.add_argument("--source", type=Path, default=HERE / "cross_domain_report.json")
    cli.add_argument("--benchmark", type=Path, default=HERE / "cross_domain_benchmark.json")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    source = json.loads(args.source.read_text())
    if source["benchmark_sha256"] != hashlib.sha256(args.benchmark.read_bytes()).hexdigest():
        raise ValueError("saved parser output and benchmark differ")
    items = source["decisions"]["columns"]
    if len(items) != 2 * len(benchmark["cases"]):
        raise ValueError("saved column count differs from benchmark")
    settings = source["decisions"]["scoring_provenance"]["retrieval"]["embedding"]
    embedder = BaseEmbedder(settings["backend"], settings["model"],
                            input_format=settings["input_format"])
    domains = defaultdict(list)
    for idx, case in enumerate(benchmark["cases"]):
        domains[case["domain"]].extend((2 * idx, 2 * idx + 1))
    retained = set()
    candidate_count = 0
    domain_counts = {}
    for domain, ids in domains.items():
        part = generate_candidates([items[i] for i in ids], embedder,
                                   threshold=args.threshold, top_k=args.top_k,
                                   representation="full_name", search_method="exact")
        if part["bases_embedded"]:
            raise RuntimeError("embedding cache miss; aborting controlled experiment")
        candidate_count += part["candidate_count"]
        domain_counts[domain] = part["candidate_count"]
        retained.update((ids[p["left_id"]], ids[p["right_id"]]) for p in part["candidates"])
    positives = [idx for idx, case in enumerate(benchmark["cases"])
                 if case["expected_relation"] == "EQUIVALENT"]
    hits = [idx for idx in positives if (2 * idx, 2 * idx + 1) in retained]
    misses = [benchmark["cases"][idx]["id"] for idx in positives if idx not in hits]
    report = dict(source=str(args.source), benchmark_sha256=source["benchmark_sha256"],
                  embedding=settings, representation="full_name", threshold=args.threshold,
                  top_k=args.top_k, positive_cases=len(positives), positive_hits=len(hits),
                  positive_retrieval_recall=len(hits) / len(positives), candidate_count=candidate_count,
                  domain_candidate_counts=domain_counts, missed_positive_cases=misses,
                  embedding_cache_hits=embedder.cache_hits, bases_embedded=embedder.embedded_count)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
