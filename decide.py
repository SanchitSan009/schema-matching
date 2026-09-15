"""Checkpoint 10: relation-led pair decisions; no blind averaging or clustering."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from candidates import HERE, add_retrieval_options, checked_items, generate_candidates, read_csv_names
from decompose import Parser
from embeddings import BaseEmbedder
from relations import RelationClassifier, validate_relations
from score import CONTEXT_FORMAT, score_candidates

DEFAULT_SUPPORT = dict(base_min=0.90, qualifier_min=0.85, lexical_min=0.80)


def validate_support(support):
    if set(support) != set(DEFAULT_SUPPORT) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
        for v in support.values()
    ):
        raise ValueError("support thresholds must be finite values in [0, 1]")


def decide_pair(signals, relation, ambiguous=False, presence_mismatch=False, support=None):
    support = DEFAULT_SUPPORT if support is None else support
    validate_support(support)
    if set(signals) != {"base", "qualifier", "lexical"} or any(
        isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1
        for s in signals.values()
    ):
        raise ValueError("base, qualifier and lexical signals must be in [0, 1]")

    def result(decision, rule):
        return dict(decision=decision, decision_rule=rule)

    if signals["base"] < support["base_min"]:
        return result("REJECT", "base_below_compatibility_gate")
    if relation is None:
        return result("UNCERTAIN", "relation_not_available")
    validate_relations({"items": [dict(relation, id=0)]}, 1)
    if relation["base_relation"] == "INCOMPATIBLE":
        return result("REJECT", "semantically_incompatible_bases")
    if relation["relation"] == "CONTRASTING":
        return result("REJECT", "contrasting_qualifiers")
    if relation["relation"] == "RELATED_BUT_DIFFERENT":
        return result("REJECT", "related_but_different_qualifiers")
    if relation["base_relation"] == "UNCERTAIN" or relation["relation"] == "UNCERTAIN":
        return result("UNCERTAIN", "semantic_relation_unresolved")
    if ambiguous or presence_mismatch:
        return result("UNCERTAIN", "decomposition_or_scope_needs_review")
    if signals["qualifier"] >= support["qualifier_min"] or signals["lexical"] >= support["lexical_min"]:
        return result("ACCEPT", "equivalent_relation_with_support")
    return result("UNCERTAIN", "equivalent_relation_without_enough_support")


def decide_report(scored, classifier, context="", support=None, fresh=False):
    from schema_context import value_evidence
    support = dict(DEFAULT_SUPPORT if support is None else support)
    validate_support(support)
    columns = checked_items(scored["columns"])
    ids = [c.get("id") for c in scored["columns"]]
    if any(type(i) is not int for i in ids) or ids != list(range(len(columns))):
        raise ValueError("scored columns must have consecutive integer IDs")
    requests, eligible, seen = [], [], set()
    plans = []
    for pair in scored["scored_candidates"]:
        i, j = pair["left_id"], pair["right_id"]
        if type(i) is not int or type(j) is not int or not 0 <= i < j < len(columns) or (i, j) in seen:
            raise ValueError("invalid or duplicate pair IDs")
        seen.add((i, j))
        left, right = columns[i], columns[j]
        ambiguous = left["ambiguous"] or right["ambiguous"]
        missing = bool(left["qualifiers"]) != bool(right["qualifiers"])
        preliminary = decide_pair(pair["signals"], None, ambiguous, missing, support)
        evidence = value_evidence(left, right)
        pair = dict(pair, value_evidence=evidence)
        plans.append((pair, left, right, ambiguous, missing, preliminary))
        if preliminary["decision"] != "REJECT":
            eligible.append(len(plans) - 1)
            request = dict(base_a=left["base"], base_b=right["base"],
                qualifiers_a=left["qualifiers"], qualifiers_b=right["qualifiers"], context=context)
            if left.get("schema") or right.get("schema"):
                request.update(schema_a=left.get("schema", {}), schema_b=right.get("schema", {}))
            if evidence["available"] or evidence["units"] != "unknown":
                request["value_evidence"] = evidence
            requests.append(request)
    predictions = classifier.classify(requests, fresh=fresh)
    if len(predictions) != len(requests):
        raise ValueError("classifier omitted pair predictions")
    relations = dict(zip(eligible, predictions))
    decisions = []
    for idx, (pair, left, right, ambiguous, missing, preliminary) in enumerate(plans):
        relation = relations.get(idx)
        final = decide_pair(pair["signals"], relation, ambiguous, missing, support)
        if final["decision"] == "ACCEPT" and pair["value_evidence"]["requires_review"]:
            final = dict(decision="UNCERTAIN", decision_rule="value_type_or_units_need_review")
        decisions.append(dict(pair, left=left["original"], right=right["original"],
                              relation=relation, **final))
    return dict(stage="pair_decisions", timestamp=datetime.now(timezone.utc).isoformat(),
                classifier=classifier.settings, context=context, support_thresholds=support,
                decision_policy="relation-led-v1", counts=dict(Counter(r["decision"] for r in decisions)),
                pair_count=len(decisions), classified_pairs=len(requests),
                scoring_provenance={k: v for k, v in scored.items() if k not in {"columns", "scored_candidates", "evaluation"}},
                columns=scored["columns"], decisions=decisions,
                limitation="Pair decisions only. No clustering, calibrated probabilities, or automatic stronger-model escalation.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("names", nargs="*")
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--scores", type=Path, help="Checkpoint 7 scored report")
    source.add_argument("--csv", type=Path)
    cli.add_argument("--column")
    cli.add_argument("--parser-model")
    cli.add_argument("--relation-model")
    cli.add_argument("--context", default="", help="Optional factual schema description")
    cli.add_argument("--fresh", action="store_true", help="Ignore cached relation judgments")
    cli.add_argument("--base-min", type=float, default=DEFAULT_SUPPORT["base_min"])
    cli.add_argument("--qualifier-min", type=float, default=DEFAULT_SUPPORT["qualifier_min"])
    cli.add_argument("--lexical-min", type=float, default=DEFAULT_SUPPORT["lexical_min"])
    cli.add_argument("--output", type=Path, default=HERE / "decision_report.json")
    add_retrieval_options(cli)
    args = cli.parse_args()
    support = dict(base_min=args.base_min, qualifier_min=args.qualifier_min, lexical_min=args.lexical_min)
    try:
        validate_support(support)
    except ValueError as exc:
        cli.error(str(exc))
    if args.names and (args.scores or args.csv):
        cli.error("provide names, --scores, or --csv exclusively")
    if args.column is not None and args.csv is None:
        cli.error("--column requires --csv")
    if not -1 <= args.threshold <= 1 or args.top_k < 1 or args.max_pairs < 1:
        cli.error("invalid retrieval settings")
    if args.scores:
        scored = json.loads(args.scores.read_text())
    else:
        names = read_csv_names(args.csv, args.column) if args.csv else args.names
        if not names:
            cli.error("provide names or a nonempty file")
        parser = Parser(args.parser_model)
        base_embedder = BaseEmbedder(args.backend, args.embedding_model, args.cache)
        candidates = generate_candidates(parser.extract(names), base_embedder, args.threshold, args.top_k, args.max_pairs)
        candidates["parser_model"] = parser.model
        qualifier_embedder = BaseEmbedder(base_embedder.backend, base_embedder.model, args.cache, input_format=CONTEXT_FORMAT)
        scored = score_candidates(candidates, qualifier_embedder)
    result = decide_report(scored, RelationClassifier(args.relation_model), args.context, support, args.fresh)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
