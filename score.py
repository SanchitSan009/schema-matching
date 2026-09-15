"""Checkpoints 6–7: contextual qualifier similarity and weighted equivalence scores."""

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import math
from pathlib import Path

import numpy as np

from candidates import HERE, add_retrieval_options, checked_items, generate_candidates, read_csv_names
from decompose import Parser
from embeddings import BaseEmbedder, unit_vectors
from normalize import normalize_column

DEFAULT_WEIGHTS = (0.25, 0.65, 0.10)
CONTEXT_FORMAT = "base-conditioned-qualifiers-v1"


def validate_weights(weights):
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("provide three finite nonnegative weights: base, qualifier, lexical")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
        raise ValueError("base, qualifier, lexical weights must sum to 1")
    return dict(zip(("base", "qualifier", "lexical"), weights))


def qualifier_text(base, qualifiers):
    # Sorting treats qualifier order as irrelevant, preserving compound boundaries
    # and multiplicity. JSON distinguishes empty lists from literal qualifier words.
    return f"Attribute property: {base}\nQualifiers: {json.dumps(sorted(qualifiers), ensure_ascii=False)}"


def isolated_text(qualifiers):
    return f"Qualifiers: {json.dumps(sorted(qualifiers), ensure_ascii=False)}"


def lexical_similarity(left, right):
    left, right = normalize_column(left), normalize_column(right)
    # SequenceMatcher can be directional: average both orders for symmetry.
    return (SequenceMatcher(None, left, right, autojunk=False).ratio()
            + SequenceMatcher(None, right, left, autojunk=False).ratio()) / 2


def score_candidates(report, embedder, weights=DEFAULT_WEIGHTS, isolated_baseline=False):
    weights = validate_weights(weights)
    columns = checked_items(report["columns"])
    raw_ids = [c.get("id") for c in report["columns"]]
    if any(type(i) is not int for i in raw_ids) or raw_ids != list(range(len(columns))):
        raise ValueError("candidate report must use consecutive column IDs starting at zero")
    if report.get("embedding") != dict(embedder.settings, input_format="bare-base-v1"):
        raise ValueError("base and qualifier embedding backend/model settings must match")
    requests = {}
    plans = []
    seen = set()
    for pair in report["candidates"]:
        i, j = pair["left_id"], pair["right_id"]
        if (type(i) is not int or type(j) is not int or not 0 <= i < j < len(columns)
                or (i, j) in seen):
            raise ValueError("candidate IDs must be valid, unique, ordered pairs")
        seen.add((i, j))
        base_score = pair["base_similarity"]
        if isinstance(base_score, bool) or not isinstance(base_score, (int, float)) or not math.isfinite(base_score) or not -1 <= base_score <= 1:
            raise ValueError("base similarity must be a finite cosine in [-1, 1]")
        left, right = columns[i], columns[j]
        contexts = sorted({left["base"], right["base"]})
        comparisons = [(base, qualifier_text(base, left["qualifiers"]),
                         qualifier_text(base, right["qualifiers"])) for base in contexts]
        for _, a, b in comparisons:
            requests[a] = None
            requests[b] = None
        baseline = (isolated_text(left["qualifiers"]), isolated_text(right["qualifiers"]))
        if isolated_baseline:
            for text in baseline:
                requests[text] = None
        plans.append((pair, left, right, comparisons, baseline))
    hits_before, encoded_before = embedder.cache_hits, embedder.embedded_count
    if requests:
        vectors = unit_vectors(embedder.encode(list(requests)), len(requests))
        requests = dict(zip(requests, vectors))

    def cosine(a, b):
        return float(np.clip(requests[a] @ requests[b], -1, 1))

    scored = []
    for pair, left, right, comparisons, baseline in plans:
        contexts = [dict(base=base, left_text=a, right_text=b, cosine=cosine(a, b))
                    for base, a, b in comparisons]
        qualifier_cosine = sum(c["cosine"] for c in contexts) / len(contexts)
        signals = dict(base=max(0.0, pair["base_similarity"]),
                       qualifier=max(0.0, qualifier_cosine),
                       lexical=lexical_similarity(left["normalized"], right["normalized"]))
        contributions = {key: weights[key] * value for key, value in signals.items()}
        result = dict(pair, left=left["original"], right=right["original"],
                      qualifier_contexts=contexts, qualifier_cosine=qualifier_cosine,
                      signals=signals, weighted_contributions=contributions,
                      equivalence_score=min(1.0, sum(contributions.values())),
                      decomposition_ambiguous=left["ambiguous"] or right["ambiguous"],
                      qualifier_presence_mismatch=bool(left["qualifiers"]) != bool(right["qualifiers"]))
        if isolated_baseline:
            result["isolated_qualifier_cosine"] = cosine(*baseline)
        scored.append(result)
    return dict(stage="experimental_equivalence_scores", timestamp=datetime.now(timezone.utc).isoformat(),
                score_interpretation="Uncalibrated weighted similarity; not a probability or merge decision",
                weights=weights, qualifier_embedding=embedder.settings,
                qualifier_cache_hits=embedder.cache_hits - hits_before,
                qualifier_texts_embedded=embedder.embedded_count - encoded_before,
                retrieval={k: v for k, v in report.items() if k not in {"columns", "candidates"}},
                columns=report["columns"], scored_candidates=scored)


def add_score_options(cli):
    cli.add_argument("--weights", type=float, nargs=3, default=DEFAULT_WEIGHTS,
                     metavar=("BASE", "QUALIFIER", "LEXICAL"), help="Nonnegative weights summing to 1")
    cli.add_argument("--isolated-baseline", action="store_true",
                     help="Also measure isolated qualifier similarity for comparison")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("names", nargs="*")
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--candidates", type=Path, help="Reuse a Checkpoint 5 candidate report")
    source.add_argument("--csv", type=Path)
    cli.add_argument("--column", help="CSV field containing names; otherwise process headers")
    cli.add_argument("--parser-model")
    cli.add_argument("--output", type=Path, default=HERE / "equivalence_report.json")
    add_retrieval_options(cli)
    add_score_options(cli)
    args = cli.parse_args()
    try:
        validate_weights(args.weights)
    except ValueError as exc:
        cli.error(str(exc))
    if args.names and (args.csv or args.candidates):
        cli.error("provide names, --csv, or --candidates exclusively")
    if args.column is not None and args.csv is None:
        cli.error("--column requires --csv")
    if not -1 <= args.threshold <= 1 or args.top_k < 1 or args.max_pairs < 1:
        cli.error("invalid retrieval settings")
    if args.candidates:
        report = json.loads(args.candidates.read_text())
        settings = report["embedding"]
        if ((args.backend and args.backend != settings["backend"]) or
                (args.embedding_model and args.embedding_model != settings["model"])):
            cli.error("saved candidates require their original embedding backend/model")
        backend, model = settings["backend"], settings["model"]
    else:
        names = read_csv_names(args.csv, args.column) if args.csv else args.names
        if not names:
            cli.error("provide column names or a nonempty input file")
        parser = Parser(args.parser_model)
        base_embedder = BaseEmbedder(args.backend, args.embedding_model, args.cache)
        report = generate_candidates(parser.extract(names), base_embedder,
                                     args.threshold, args.top_k, args.max_pairs)
        report["parser_model"] = parser.model
        backend, model = base_embedder.backend, base_embedder.model
    qualifier_embedder = BaseEmbedder(backend, model, args.cache, input_format=CONTEXT_FORMAT)
    result = score_candidates(report, qualifier_embedder, args.weights, args.isolated_baseline)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Scored {len(result['scored_candidates'])} candidate pairs. Full results: {args.output}")


if __name__ == "__main__":
    main()
