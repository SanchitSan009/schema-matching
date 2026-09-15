import argparse
import json
from pathlib import Path

from candidates import HERE, generate_candidates, read_csv_names, checked_items
from decompose import Parser
from embeddings import BaseEmbedder
from score import CONTEXT_FORMAT, score_candidates
from relations import RelationClassifier
from decide import decide_report
from calibrate import confidence_report
from cluster_confirmed import cluster_confirmed
from schema_context import enrich_records, csv_records, metadata, profile_values


def prepare_records(records, parser):
    records = enrich_records(records)
    missing = [i for i, r in enumerate(records) if "base" not in r or "qualifiers" not in r]
    parsed = parser.extract([records[i]["name"] for i in missing],
                            [metadata(records[i]) for i in missing]) if missing else []
    predictions = dict(zip(missing, parsed))
    items = []
    for i, r in enumerate(records):
        item = predictions.get(i) or dict(original=r["name"], base=r["base"], qualifiers=r["qualifiers"],
                                          ambiguous=r.get("ambiguous", False), reason="Supplied decomposition")
        item = dict(item, schema=metadata(r))
        if "values" in r:
            item["value_profile"] = profile_values(r["values"])
            item["value_profile"]["truncated"] |= r.get("values_truncated", False)
        items.append(item)
    return checked_items(items)


def pipeline(records, parser=None, embedder=None, classifier=None, top_k=15,
             representation="full_name", method="auto", stop_after="clusters", calibration=None):
    parser, embedder = parser or Parser(), embedder or BaseEmbedder()
    items = prepare_records(records, parser)
    candidates = generate_candidates(items, embedder, top_k=top_k, representation=representation, search_method=method)
    candidates["parser_model"] = parser.model
    if stop_after == "retrieval":
        return candidates
    qualifier_embedder = BaseEmbedder(embedder.backend, embedder.model, embedder.cache, input_format=CONTEXT_FORMAT)
    scores = score_candidates(candidates, qualifier_embedder)
    decisions = decide_report(scores, classifier or RelationClassifier())
    decisions["evidence_version"] = "schema-values-v1"
    if stop_after == "decisions":
        return decisions
    bands = confidence_report(decisions, calibration)
    result = cluster_confirmed(bands)
    result["confidence"] = bands
    return result


def main():
    cli = argparse.ArgumentParser()
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--schema", type=Path)
    source.add_argument("--csv", type=Path)
    cli.add_argument("--column")
    cli.add_argument("--profile-values", action="store_true")
    cli.add_argument("--top-k", type=int, default=15)
    cli.add_argument("--representation", choices=["base", "full_name"], default="full_name")
    cli.add_argument("--search", choices=["auto", "exact", "hnsw"], default="auto")
    cli.add_argument("--stop-after", choices=["retrieval", "decisions", "clusters"], default="clusters")
    cli.add_argument("--calibration", type=Path)
    cli.add_argument("--output", type=Path, default=HERE / "pipeline_report.json")
    args = cli.parse_args()
    if args.top_k < 1 or args.column is not None and args.csv is None:
        cli.error("invalid top-k or CSV column option")
    if args.schema:
        payload = json.loads(args.schema.read_text())
        records = payload if isinstance(payload, list) else payload["columns"]
    elif args.column:
        if args.profile_values:
            cli.error("value profiling requires data columns, not an attribute-name list")
        records = [dict(name=n) for n in read_csv_names(args.csv, args.column)]
    else:
        records = csv_records(args.csv, args.profile_values)
    if not records:
        cli.error("input must contain columns")
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    result = pipeline(records, top_k=args.top_k, representation=args.representation, method=args.search,
                      stop_after=args.stop_after, calibration=calibration)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
