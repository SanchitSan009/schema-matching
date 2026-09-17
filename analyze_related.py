"""Analyze RELATED_BUT_DIFFERENT decisions and test base-gate removal."""

import argparse
import collections
import json
import math
from itertools import combinations
from pathlib import Path

from decide import DEFAULT_SUPPORT, decide_pair

HERE = Path(__file__).resolve().parent


def load_gold_pairs(path, col_names):
    """Load gold pairs using the decision-report column ordering for ID mapping."""
    groups = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label, values = line.split(":", 1)
        groups.append({"id": label.strip(), "members": [v.strip() for v in values.split(",")]})
    name_to_id = {n: i for i, n in enumerate(col_names)}
    gold_pairs = set()
    for group in groups:
        for a, b in combinations(group["members"], 2):
            gold_pairs.add(tuple(sorted((name_to_id[a], name_to_id[b]))))
    return gold_pairs, groups


def analyze(decision_path, gold_path):
    report = json.loads(decision_path.read_text())
    decisions = report["decisions"]
    columns = report["columns"]

    names = [c["original"] for c in columns]
    gold_pairs, gold_groups = load_gold_pairs(gold_path, names)

    rbd = [d for d in decisions if d.get("relation", {}).get("relation") == "RELATED_BUT_DIFFERENT"]
    rbd_set = {(d["left_id"], d["right_id"]) for d in rbd}

    # Gold pairs that are RELATED_BUT_DIFFERENT
    rbd_gold = rbd_set & gold_pairs
    rbd_nongold = rbd_set - gold_pairs

    # Gold pairs missed overall
    all_pairs = {(d["left_id"], d["right_id"]) for d in decisions}
    gold_in_scope = gold_pairs & all_pairs
    gold_missed = gold_pairs - all_pairs

    # GOLD pairs where Gemini said RELATED_BUT_DIFFERENT (false negatives)
    gold_rbd = [d for d in rbd if (d["left_id"], d["right_id"]) in gold_pairs]

    print("=" * 70)
    print("RELATED_BUT_DIFFERENT DECISION ANALYSIS")
    print("=" * 70)
    print(f"Total decisions: {len(decisions)}")
    print(f"RELATED_BUT_DIFFERENT: {len(rbd)}")
    print(f"  - Gold pairs misclassified: {len(gold_rbd)}")
    print(f"  - Non-gold pairs correctly rejected: {len(rbd_set) - len(gold_rbd)}")
    print(f"Gold pairs in scope: {len(gold_in_scope)}")
    print(f"Gold pairs outside scope (retrieval miss): {len(gold_missed)}")
    print()

    # Signal profiles of RELATED_BUT_DIFFERENT decisions
    print("--- SIGNAL PROFILES OF RELATED_BUT_DIFFERENT DECISIONS ---")
    base_sims = [d["signals"]["base"] for d in rbd]
    qual_sims = [d["signals"]["qualifier"] for d in rbd]
    lex_sims = [d["signals"]["lexical"] for d in rbd]
    print(f"  base similarity:    min={min(base_sims):.4f}  max={max(base_sims):.4f}  "
          f"median={sorted(base_sims)[len(base_sims)//2]:.4f}  mean={sum(base_sims)/len(base_sims):.4f}")
    print(f"  qualifier cosine:   min={min(qual_sims):.4f}  max={max(qual_sims):.4f}  "
          f"median={sorted(qual_sims)[len(qual_sims)//2]:.4f}  mean={sum(qual_sims)/len(qual_sims):.4f}")
    print(f"  lexical similarity: min={min(lex_sims):.4f}  max={max(lex_sims):.4f}  "
          f"median={sorted(lex_sims)[len(lex_sims)//2]:.4f}  mean={sum(lex_sims)/len(lex_sims):.4f}")
    print()

    # Group by base: which base pairs dominate RELATED_BUT_DIFFERENT
    base_groups = collections.Counter()
    for d in rbd:
        lbase = columns[d["left_id"]]["base"]
        rbase = columns[d["right_id"]]["base"]
        key = tuple(sorted([lbase, rbase]))
        base_groups[key] += 1
    print("--- TOP BASE-PAIR GROUPS IN RELATED_BUT_DIFFERENT ---")
    for (b1, b2), count in base_groups.most_common(20):
        in_gold = sum(1 for d in rbd
                      if tuple(sorted([columns[d["left_id"]]["base"], columns[d["right_id"]]["base"]])) == (b1, b2)
                      and (d["left_id"], d["right_id"]) in gold_pairs)
        print(f"  {b1} / {b2}: {count} decisions ({in_gold} gold)")
    print()

    # Qualifier patterns in RELATED_BUT_DIFFERENT: same base, different qualifier structure
    print("--- QUALIFIER PATTERNS IN GOLD RBD (false negatives) ---")
    for d in sorted(gold_rbd, key=lambda x: (-x["signals"]["qualifier"], -x["signals"]["lexical"])):
        l = columns[d["left_id"]]
        r = columns[d["right_id"]]
        print(f"  [{d['left_id']:3d}] {l['original']:35s}  vs  [{d['right_id']:3d}] {r['original']:35s}  "
              f"base={d['signals']['base']:.3f}  qual={d['signals']['qualifier']:.3f}  "
              f"lex={d['signals']['lexical']:.3f}")
        print(f"        left  base={l['base']!r}  quals={l['qualifiers']}")
        print(f"        right base={r['base']!r}  quals={r['qualifiers']}")
    print()

    # Same-base RELATED_BUT_DIFFERENT (most likely to be wrong)
    same_base_rbd = [d for d in rbd if columns[d["left_id"]]["base"] == columns[d["right_id"]]["base"]]
    same_base_gold_rbd = [d for d in same_base_rbd if (d["left_id"], d["right_id"]) in gold_pairs]
    print(f"--- SAME-BASE RELATED_BUT_DIFFERENT ---")
    print(f"  Same-base RBD: {len(same_base_rbd)} ({len(same_base_gold_rbd)} gold)")
    if same_base_rbd:
        squal = [d["signals"]["qualifier"] for d in same_base_rbd]
        print(f"  qualifier cosine: min={min(squal):.4f} max={max(squal):.4f} "
              f"mean={sum(squal)/len(squal):.4f}")
    print()

    # Different-base RELATED_BUT_DIFFERENT
    diff_base_rbd = [d for d in rbd if columns[d["left_id"]]["base"] != columns[d["right_id"]]["base"]]
    diff_base_gold_rbd = [d for d in diff_base_rbd if (d["left_id"], d["right_id"]) in gold_pairs]
    print(f"--- DIFFERENT-BASE RELATED_BUT_DIFFERENT ---")
    print(f"  Diff-base RBD: {len(diff_base_rbd)} ({len(diff_base_gold_rbd)} gold)")
    if diff_base_gold_rbd:
        for d in diff_base_gold_rbd:
            l = columns[d["left_id"]]
            r = columns[d["right_id"]]
            print(f"  [{d['left_id']}] {l['original']} (base={l['base']!r})  vs  "
                  f"[{d['right_id']}] {r['original']} (base={r['base']!r})  "
                  f"qual={d['signals']['qualifier']:.3f}  lex={d['signals']['lexical']:.3f}")
    print()

    # All gold-pair misses: classify by cause
    print("=" * 70)
    print("GOLD PAIR MISS BREAKDOWN")
    print("=" * 70)
    causes = collections.Counter()
    for pair in sorted(gold_missed):
        causes["retrieval_miss"] += 1
    for d in decisions:
        if (d["left_id"], d["right_id"]) in gold_pairs:
            if d["decision"] == "REJECT":
                causes[f"rejected_{d['decision_rule']}"] += 1
            elif d["decision"] == "UNCERTAIN":
                causes[f"uncertain_{d['decision_rule']}"] += 1
            else:
                causes["accepted"] += 1
    for cause, count in causes.most_common():
        print(f"  {cause}: {count}")
    print()

    # BASE GATE ANALYSIS
    print("=" * 70)
    print("BASE COMPATIBILITY GATE ANALYSIS")
    print("=" * 70)
    support = dict(DEFAULT_SUPPORT)
    gated_count = 0
    gated_gold = 0
    gated_details = []
    for d in decisions:
        preliminary = decide_pair(d["signals"], None, False, False, support)
        if preliminary["decision"] == "REJECT" and preliminary["decision_rule"] == "base_below_compatibility_gate":
            gated_count += 1
            is_gold = (d["left_id"], d["right_id"]) in gold_pairs
            if is_gold:
                gated_gold += 1
            gated_details.append((d, is_gold))
    print(f"  Currently gated (base < {support['base_min']}): {gated_count} ({gated_gold} gold)")
    if gated_details:
        print("  Gated pairs:")
        for d, is_gold in gated_details[:30]:
            l = columns[d["left_id"]]
            r = columns[d["right_id"]]
            print(f"    {'*' if is_gold else ' '} [{d['left_id']}] {l['original']} (base={d['signals']['base']:.4f})  vs  "
                  f"[{d['right_id']}] {r['original']} (base={d['signals']['base']:.4f})")
    print()

    # What would happen if base gate was removed?
    # Re-run all decisions without the base gate
    print("--- WHAT-IF: BASE GATE REMOVED ---")
    no_gate_support = dict(support, base_min=0.0)
    new_accepts = 0
    new_rejects = 0
    new_uncertain = 0
    newly_reached_gemini = 0
    for d in decisions:
        # Original decision with base gate
        original = decide_pair(d["signals"], d.get("relation"), False, False, support)
        # Decision without base gate
        no_gate = decide_pair(d["signals"], d.get("relation"), False, False, no_gate_support)
        if original["decision"] != no_gate["decision"]:
            newly_reached_gemini += 1
            if no_gate["decision"] == "ACCEPT":
                new_accepts += 1
            elif no_gate["decision"] == "REJECT":
                new_rejects += 1
            else:
                new_uncertain += 1
    print(f"  Pairs whose decision changes: {newly_reached_gemini}")
    print(f"    New ACCEPTs: {new_accepts}")
    print(f"    New REJECTs: {new_rejects}")
    print(f"    New UNCERTAIN: {new_uncertain}")
    print()

    # Show pairs that would become ACCEPT
    if new_accepts > 0:
        print("  Pairs that would become ACCEPT:")
        for d in decisions:
            original = decide_pair(d["signals"], d.get("relation"), False, False, support)
            no_gate = decide_pair(d["signals"], d.get("relation"), False, False, no_gate_support)
            if original["decision"] != no_gate["decision"] and no_gate["decision"] == "ACCEPT":
                l = columns[d["left_id"]]
                r = columns[d["right_id"]]
                is_gold = (d["left_id"], d["right_id"]) in gold_pairs
                print(f"    {'*' if is_gold else ' '} [{d['left_id']}] {l['original']}  vs  "
                      f"[{d['right_id']}] {r['original']}  base={d['signals']['base']:.4f}  "
                      f"qual={d['signals']['qualifier']:.4f}  lex={d['signals']['lexical']:.4f}  "
                      f"relation={d.get('relation', {}).get('relation')}")
    print()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--decisions", type=Path,
                     default=HERE / "benchmark250_decision_report.json")
    cli.add_argument("--gold", type=Path,
                     default=HERE / "benchmark250_gold_clusters.txt")
    args = cli.parse_args()
    analyze(args.decisions, args.gold)


if __name__ == "__main__":
    main()
