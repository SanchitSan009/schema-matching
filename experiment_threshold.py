"""Compare retrieval at threshold 0.90 vs 0.88 on the 250-column benchmark."""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from candidates import generate_candidates, checked_items
from embeddings import BaseEmbedder

HERE = Path(__file__).resolve().parent


def load_gold_pairs(col_names):
    groups = []
    for line in (HERE / "benchmark250_gold_clusters.txt").read_text().splitlines():
        if not line.strip():
            continue
        label, values = line.split(":", 1)
        groups.append([v.strip() for v in values.split(",")])
    name_to_id = {n: i for i, n in enumerate(col_names)}
    pairs = set()
    for g in groups:
        for a, b in combinations(g, 2):
            pairs.add(tuple(sorted((name_to_id[a], name_to_id[b]))))
    return pairs


def run_experiment(threshold, top_k, items, embedder, gold_pairs, col_names):
    report = generate_candidates(items, embedder, threshold=threshold, top_k=top_k,
                                 representation="full_name", search_method="exact")
    candidate_ids = {(p["left_id"], p["right_id"]) for p in report["candidates"]}
    gold_recall = len(candidate_ids & gold_pairs)
    total_gold = len(gold_pairs)

    # Check base-similarity distribution for candidates
    base_sims = [p["base_similarity"] for p in report["candidates"]]
    below_090 = [p for p in report["candidates"] if p["base_similarity"] < 0.90]

    print(f"\n{'='*60}")
    print(f"Threshold={threshold}  Top_k={top_k}")
    print(f"{'='*60}")
    print(f"  Candidates:          {report['candidate_count']}")
    print(f"  Gold pair recall:    {gold_recall}/{total_gold} ({gold_recall/total_gold:.1%})")
    print(f"  Gold in candidates:  {len(candidate_ids & gold_pairs)}")
    print(f"  Gold NOT in scope:   {total_gold - len(candidate_ids & gold_pairs)}")
    print(f"  Base sim range:      [{min(base_sims):.4f}, {max(base_sims):.4f}]")
    print(f"  Candidates base<0.90: {len(below_090)}")
    if below_090:
        print(f"    These would be blocked by base gate (base_min=0.90):")
        for p in below_090[:20]:
            l = col_names[p["left_id"]]
            r = col_names[p["right_id"]]
            is_gold = (p["left_id"], p["right_id"]) in gold_pairs
            print(f"      {'*' if is_gold else ' '} {l} <-> {r}  base={p['base_similarity']:.4f}")

    # Show which gold pairs are newly retrieved
    return candidate_ids, below_090


def main():
    report = json.loads((HERE / "benchmark250_retrieval_report.json").read_text())
    items = checked_items(report["columns"])
    col_names = [c["original"] for c in report["columns"]]
    gold_pairs = load_gold_pairs(col_names)
    embedder = BaseEmbedder(report["embedding"]["backend"], report["embedding"]["model"],
                            input_format=report["embedding"]["input_format"])

    # Ensure all base embeddings are cached
    bases = sorted({item["base"] for item in items})
    embedder.encode(bases)

    ids_090, below_090 = run_experiment(0.90, 15, items, embedder, gold_pairs, col_names)
    ids_088, below_088 = run_experiment(0.88, 15, items, embedder, gold_pairs, col_names)

    # Delta
    new_candidates = ids_088 - ids_090
    new_gold = (ids_088 & gold_pairs) - (ids_090 & gold_pairs)
    print(f"\n{'='*60}")
    print(f"DELTA: threshold 0.90 -> 0.88")
    print(f"{'='*60}")
    print(f"  New candidates:    {len(new_candidates)}")
    print(f"  New gold pairs:    {len(new_gold)}")
    print(f"  New base<0.90:     {len(below_088)}")
    if new_gold:
        print("  Newly recovered gold pairs:")
        for i, j in sorted(new_gold):
            print(f"    {col_names[i]} <-> {col_names[j]}")

    # Now simulate base-gate removal at 0.88
    # Candidates with base < 0.90 would be gated but not if gate removed
    gated_gold = [(p["left_id"], p["right_id"]) for p in below_088
                  if (p["left_id"], p["right_id"]) in gold_pairs]
    print(f"\n  Gold pairs gated at 0.88 (base<0.90): {len(gated_gold)}")
    for i, j in gated_gold:
        p = next(p for p in below_088 if p["left_id"] == i and p["right_id"] == j)
        print(f"    {col_names[i]} <-> {col_names[j]}  base={p['base_similarity']:.4f}")


if __name__ == "__main__":
    main()
