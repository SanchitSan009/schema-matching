# Implementation checkpoints

All checkpoint implementations, verification files, and documentation live in
`schema_matching/`. Future checkpoints will extend this directory. Configuration
continues to load from the project root `.env`.

## Checkpoint 1 — Normalize column names

Requirement: make downstream parsing consistent across camelCase, PascalCase,
snake_case, hyphens, casing, and obvious spacing issues, without domain dictionaries.

Implemented in `normalize.py`:
- Split camel/Pascal case and acronym boundaries such as `HTTPStatusCode`.
- Replace underscores and Unicode dash punctuation with spaces.
- Lowercase, trim, collapse whitespace, and canonically normalize Unicode (NFC).
- Retain digits and other punctuation, including unit symbols.
- Return original/normalized pairs as JSON; retain duplicates and empty results.
- Accept names directly, CSV headers, or a CSV field containing attribute names.
- Load the root `.env` without overriding existing environment variables. This
  checkpoint runs locally and requires no credentials or API calls.

Spelling correction is deferred: no reliable domain-independent rule currently
distinguishes a typo from a valid technical term. Abbreviations and semantic
synonyms remain unchanged for later checkpoints. Case boundary detection covers
ASCII identifier conventions; other Unicode text is retained and lowercased.
Normalization is intentionally lossy for casing/separators, so original names
remain available. A matching normalized string is not proof of equivalence.

### Run from the project root

The existing `.venv` already includes `python-dotenv`, the only external dependency
for this checkpoint.

```bash
.venv/bin/python schema_matching/normalize.py InputVoltage input_voltage input-voltage
```

All three records should have `"normalized": "input voltage"`.

For the existing CSV containing an `Attribute` field (header is case-sensitive):

```bash
.venv/bin/python schema_matching/normalize.py --csv attributes.csv --column Attribute
```

Omit `--column` to normalize CSV headers. Redirect stdout to save JSON if needed.

### Verify

```bash
.venv/bin/python -m unittest discover -s schema_matching -p 'test_*.py' -v
```

Expected: three passing tests, covering formatting examples, idempotence,
preservation of semantic distinctions, and rejection of non-string input.

Existing experimental pipelines outside this directory remain unchanged.

## Checkpoint 2 — Extract base and qualifiers

Implemented in `decompose.py`. Gemini receives normalized names and a general
decomposition instruction, without domain dictionaries or benchmark labels.
Each output retains original and normalized names, base, qualifiers, an ambiguity
flag, and a brief reason. Compound bases and compound qualifiers are permitted.
Response validation checks IDs, required fields, and exact source-word coverage.
Malformed responses fail explicitly; they are never replaced with rule-based guesses.
Empty names are rejected. Inputs are batched in groups of at most 20.

Annotation convention: preserve source wording, including abbreviations, and
partition all normalized words exactly once. Do not canonicalize synonyms yet.
This keeps decomposition distinct from later semantic base matching. Ambiguity
flags are model judgments, not calibrated probabilities.

```bash
.venv/bin/python schema_matching/decompose.py input_voltage front_left_wheel_position
```

The script loads the project root `.env` regardless of the working directory.
It uses `GEMINI_API_KEY` and defaults to `gemini-2.5-flash`, matching the project's
existing model. Override using `SCHEMA_MATCHING_MODEL` in `.env` or `--model`.
This command sends column names to Gemini and requires network access.
Dependencies are recorded in `requirements.txt` and already available in `.venv`.

## Checkpoint 3 — Validate decomposition

`benchmark.json` contains 25 manually authored starting labels spanning electrical,
mechanical, business, weather, medical, and other attributes, including the user's
examples. These are evaluation data, never runtime mappings. Labels are provisional:
compound boundaries can be ambiguous and should be reviewed against your intended
annotation convention. Formatting variants are separate cases and inflate the
weight of the input-voltage concept. This is a smoke benchmark, not evidence of
domain-independent generalization or a held-out test set.

`evaluate.py` runs extraction and writes `benchmark_report.json` with:
- Exact base accuracy.
- Qualifier accuracy using order-independent multiset equality (compound boundaries
  still matter, and duplicates are counted).
- Exact decomposition accuracy, requiring both fields to match.
- Ambiguity count and every expected/actual result, including failures.
- Model name, UTC timestamp, benchmark hash, and prompt hash.

Ambiguous predictions remain in the denominator. A successful command means the
evaluation completed, not that every decomposition passed. Each run makes fresh
API calls; results can vary even with temperature zero.

Run both extraction and benchmark evaluation in one command:

```bash
.venv/bin/python schema_matching/evaluate.py
```

Verify offline structure validation, metric calculations, and normalization:

```bash
.venv/bin/python -m unittest discover -s schema_matching -p 'test_*.py' -v
```

Expected: six passing tests. These offline tests verify the implementation;
only the live benchmark measures model decomposition quality. Checkpoints 2–3
stop at decomposition; retrieval is added separately in Checkpoints 4–5 below.

### Initial live verification

The first Gemini run achieved 21/25 (84%) for base, qualifier, and exact
decomposition accuracy. All six offline tests passed. The four mismatches were
`employee_start_date`, `order_total_amount`, `patient_body_mass`, and
`average_wind_speed`: the model included one expected qualifier in a compound
base. It flagged all four as ambiguous. Both user-supplied decomposition examples
matched. Full predictions and explanations are in `benchmark_report.json`.
The labels were not adjusted after observing predictions. This result highlights
the need to settle compound-base granularity before interpreting accuracy as
semantic correctness; it does not establish general reliability.

## Checkpoint 4 — Semantic base matching

`embeddings.py` encodes distinct extracted base strings and L2-normalizes the
vectors. No synonyms, abbreviations, or domain mappings are hardcoded. SQLite
caching in `.cache/bases.sqlite3` avoids repeated embedding calls; cache keys
include backend, model, task/input convention, and base text. Changing a model
identifier isolates its cache. A provider can update a model under the same ID;
use a fresh `--cache` file when intentionally refreshing embeddings.

`candidates.py::retrieve_bases` performs blocked cosine nearest-neighbor search.
Defaults: cosine threshold **0.90** and **5** other bases per base. A pair is
retained when either base retrieves the other above the threshold. These are
provisional Gemini retrieval settings, not equivalence probabilities. Different
models need separate threshold evaluation. Exact-base groups bypass top-k limits.

The default backend is `gemini` with `gemini-embedding-001`, using the root `.env`
and `GEMINI_API_KEY`. This makes the current implementation runnable with existing
dependencies. `SCHEMA_EMBEDDING_BACKEND` and `SCHEMA_EMBEDDING_MODEL` can configure
defaults; `--backend` and `--embedding-model` override them.

The optional `sentence-transformers` backend supports the proposed BGE-M3 and
lighter MiniLM comparison, on CPU. Install its dependencies first:

```bash
.venv/bin/python -m pip install -r schema_matching/requirements-local.txt
```

Then run the same isolated retrieval benchmark with either model:

```bash
.venv/bin/python schema_matching/evaluate_candidates.py --gold-bases --backend sentence-transformers --embedding-model BAAI/bge-m3 --threshold 0.65 --output schema_matching/bge_report.json
.venv/bin/python schema_matching/evaluate_candidates.py --gold-bases --backend sentence-transformers --embedding-model sentence-transformers/all-MiniLM-L6-v2 --threshold 0.65 --output schema_matching/minilm_report.json
```

The example local cutoff of 0.65 is only a starting point and is unverified.
First use downloads model weights into `.cache/models/`; subsequent cached runs
reuse them. Local model quality has not yet been verified in this environment.
Gemini is the initial verified implementation, not a benchmark-based model choice.
Provider references: [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
and [SentenceTransformer encoding](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

## Checkpoint 5 — Runtime candidate pairs

`candidates.py` joins column IDs through exact-base groups and retrieved semantic
base pairs. It generates unique unordered pairs, preserves duplicate column
occurrences by ID, and carries qualifiers and decomposition ambiguity forward.
`input_voltage` and `output_voltage` deliberately remain candidates: qualifier
comparison is not implemented yet. Likewise, speed/velocity retrieval does not
assert semantic equivalence in every context.

Run normalization, LLM decomposition, semantic retrieval, and pair generation:

```bash
.venv/bin/python schema_matching/candidates.py input_voltage incoming_voltage output_voltage manufacturer ambient_temp ambient_temperature vehicle_speed vehicle_velocity
```

For the existing attribute-list CSV:

```bash
.venv/bin/python schema_matching/candidates.py --csv attributes.csv --column Attribute
```

Omit `--column` to process actual CSV headers. Reuse prior LLM output without
calling decomposition again:

```bash
.venv/bin/python schema_matching/candidates.py --decompositions schema_matching/benchmark_report.json
```

Saved inputs must retain `original`, `base`, `qualifiers`, `ambiguous`, and `reason`.
The CLI accepts decomposition lists, `decompose.py` JSON, prior candidate reports,
or Checkpoint 3 reports
(only their `actual` predictions are used). Runtime never reads benchmark labels.

Output defaults to `candidate_report.json`; change it with `--output`. The report
includes columns, base neighbors, base cosine scores, pair reasons, cache counts,
candidate counts, and reduction relative to all possible column pairs.

Scalability limits: similarity blocks use O(256 × unique bases) memory and exact
search remains quadratic in unique bases. Column pairs are expanded only after
retrieval. Very large same-base groups can still create quadratic output; the
default `--max-pairs 100000` fails explicitly before expansion instead of silently
dropping candidates. An ANN index or streamed expansion is future work.

### Verify Checkpoints 4–5

```bash
.venv/bin/python -m unittest discover -s schema_matching -p 'test_*.py' -v
.venv/bin/python schema_matching/evaluate_candidates.py
```

Offline expectation: 13 passing tests across all checkpoints. Synthetic vectors
test retrieval mechanics, caching, pair identity, bounds, and metric accounting;
they do not demonstrate semantic model quality.

`candidate_benchmark.json` provides 12 manual evaluation columns and exhaustive
retrieval-group labels (6 positive pairs, 60 negative pairs). The evaluator sends
only names to the live parser; group labels are used only after retrieval to
calculate precision, recall, misses, and unexpected pairs. `--gold-bases` instead
isolates retrieval from decomposition errors. Results go to
`candidate_benchmark_report.json`. Success means the run completed, not that
quality passed a threshold. This small fixture is not a held-out generalization
test, and the default threshold has not been calibrated across domains.

Checkpoints 4–5 end at runtime candidates. Qualifier comparison and experimental
scoring are added in Checkpoints 6–7 below; clustering remains unimplemented.

### Initial Checkpoint 4–5 verification results

- All 13 offline tests passed.
- At the original 0.65 cutoff, Gemini retrieved 53/66 pairs: 6 true positives,
  47 false positives, and no misses. This result is preserved in
  `candidate_benchmark_initial_report.json`.
- Cached base inspection showed unrelated similarities around 0.82–0.88,
  `temp`/`temperature` at 0.9203, and `speed`/`velocity` at 0.9553.
- The default was then raised to 0.90. A fresh live parser run with cached
  embeddings retrieved 6/66 pairs, all 6 expected candidates, no unexpected pairs,
  and 90.9% pair reduction. See `candidate_benchmark_report.json`.
- Isolated retrieval using manual bases produced the same metrics, saved in
  `candidate_benchmark_gold_report.json`. Reusing the initial live decompositions
  through the runtime CLI also produced the same six pairs in `candidate_report.json`.

The cutoff was selected after observing this benchmark. Consequently its 100%
precision/recall is a development-fixture result, not independent validation.
No benchmark labels or pair-specific exceptions were added to runtime retrieval.
Local BGE-M3/MiniLM adapters are provided but their models were not installed or
run during this checkpoint; their comparison remains outstanding.

## Checkpoint 6 — Compare qualifiers conditioned on the base

Implemented in `score.py`. Each qualifier list is embedded in this versioned,
domain-independent text format:

```text
Attribute property: voltage
Qualifiers: ["input"]
```

Compare this with the same property and `["incoming"]` or `["output"]`. All
qualifiers participate, including location/entity modifiers. Qualifiers are sorted
for order independence; compound strings and duplicates are retained. Empty
qualifiers are represented explicitly as `[]`; missing versus present qualifiers
also produce a diagnostic flag. No missing qualifier is silently inferred.

For different retrieved bases, compare the lists in each shared base context and
average the two cosine similarities. For example, with `temp`/`temperature`, both
lists are compared under `temp`, then both under `temperature`. This is symmetric
and avoids comparing a different property on each side within one cosine.

The report exposes the exact context strings, per-context cosine, and averaged
qualifier cosine. `--isolated-baseline` also embeds qualifier lists without any
base text for comparison. `embeddings.py` now accepts a cache input-format namespace
so these embeddings are isolated from the bare-base cache. No domain rules,
antonym vetoes, cross-encoder, or LLM pair matcher are used.

## Checkpoint 7 — Weighted equivalence score

`score.py` combines three signals:

```text
score = 0.25 × Sbase + 0.65 × Squalifier + 0.10 × Slexical
```

Base and qualifier cosines are clipped at zero for the weighted sum, keeping
signals in [0, 1]; their raw cosines remain available for inspection. Lexical
similarity uses the average of both directions of Python's `SequenceMatcher` on
normalized full column names, with autojunk disabled. It can support spelling
variants without correcting or replacing words. Weights are provisional, with
more weight on qualifiers; override with `--weights 0.25 0.65 0.10`. Weights must
be finite, nonnegative, and sum to one.

Outputs include individual signals, weighted contributions, the final score,
decomposition ambiguity, and qualifier-presence mismatch. The score is an
**uncalibrated similarity score**, not a probability of equivalence. Runtime does
not label pairs as equivalent or merge columns. High base/lexical similarity can
still inflate scores for non-equivalent attributes, and common base context can
inflate qualifier similarity. This is the experiment being measured.

### Run

Score a saved candidate report in one command:

```bash
.venv/bin/python schema_matching/score.py --candidates schema_matching/candidate_report.json
```

Or run normalization through scoring directly on a CSV:

```bash
.venv/bin/python schema_matching/score.py --csv attributes.csv --column Attribute
```

Positional names are also supported. Omit `--column` to read CSV headers. Output
defaults to `equivalence_report.json`; override with `--output`. All configuration
continues to use the root `.env`. Saved candidates retain their embedding model
for qualifier scoring; mixing model scores is rejected. Candidates not retrieved
at Checkpoint 5 receive no score, so lexical similarity cannot rescue an excluded
typo pair at this stage. Retrieval settings apply only to fresh input, not saved
candidate reports.

### Verify the core experiment

```bash
.venv/bin/python -m unittest discover -s schema_matching -p 'test_*.py' -v
.venv/bin/python schema_matching/evaluate_scores.py
```

Expected offline result: 20 passing tests. They cover shared-base conditioning,
score arithmetic/symmetry, lexical spelling similarity, empty/multiple qualifiers,
cache compatibility, validation, and evaluation accounting. They do not claim
semantic quality from synthetic vectors.

`equivalence_benchmark.json` defines 20 columns and 15 labeled comparisons,
including direction contrasts, wheel/tyre position, spelling variants, and missing
qualifiers. These are provisional name-only assumptions, not universal meanings.
By default evaluation uses manually supplied decomposition to isolate the
embedding experiment, then generates runtime candidates automatically. The pair
labels are used only afterward for evaluation. `--live-parser` additionally
exercises Gemini decomposition. Output is `equivalence_benchmark_report.json`.

The evaluator records retrieval misses separately, compares contextual versus
isolated qualifier ROC AUC on retained labeled pairs, and checks whether synonym
pairs outrank opposite-direction/location pairs. It also reports precision/recall
at a provisional `--diagnostic-cutoff 0.9`; this is solely an evaluation cutoff,
not a runtime merge threshold. Unretrieved positive labels count as false negatives
in those end-to-end metrics. Unlabeled runtime pairs are not assumed negative.
All scores and ranking failures remain visible; completing the command does not
mean the semantic hypothesis passed. No clustering is included.

### Initial Checkpoint 6–7 results

All 20 offline tests passed. Live Gemini embeddings with manual benchmark
decompositions retained 14/15 labeled pairs, including all seven positive pairs;
the excluded pair was `input_voltage`/`manufacturer`.

| Comparison | Contextual qualifier cosine | Weighted score |
| --- | ---: | ---: |
| input_voltage / incoming_voltage (positive label) | 0.9593 | 0.9425 |
| input_voltage / output_voltage (negative label) | 0.9625 | 0.9571 |
| wheel_position / tyre_position (positive label) | 0.9427 | 0.9368 |

At the diagnostic cutoff of 0.90, there were seven true positives and seven false
positives (50% precision, 100% recall). Contextual and isolated qualifier ROC AUC
were both 0.7143 on retained labels; weighted-score AUC was 0.6531. The voltage
and current direction rankings failed; the wheel/tyre versus left/right ranking
passed. The scores, labels, defaults, and cutoff were not adjusted to hide these
failures. Full evidence is in `equivalence_benchmark_report.json`.

Conclusion for this model and template: shared-base embedding similarity does
not reliably distinguish qualifier equivalence from relatedness. Since base
scores tie and both other signals favor input/output over input/incoming,
nonnegative reweighting alone cannot reverse that example. Other models, richer
context, or a stronger pair matcher need separate experiments before these scores
can support equivalence decisions. These results do not reject every possible
base-conditioned representation. BGE-M3/MiniLM comparison, calibration, a stronger
matcher, and clustering were outstanding at Checkpoint 7. The zero-shot matcher
and decision policy are implemented below; local embedding comparisons,
calibration, and clustering remain outstanding.

## Checkpoint 8 — Pairwise qualifier relation classifier

`relations.py` uses zero-shot Gemini classification, with no examples, training,
domain dictionary, or benchmark labels in the prompt. It compares complete
qualifier lists conditioned on both base properties and optional factual schema
context. It returns `EQUIVALENT`, `CONTRASTING`, `RELATED_BUT_DIFFERENT`, or
`UNCERTAIN`, with an explanation. It separately judges bases as `COMPATIBLE`,
`INCOMPATIBLE`, or `UNCERTAIN` to avoid treating related properties as identical.

```bash
.venv/bin/python schema_matching/relations.py --base voltage --qualifier-a input --qualifier-b output
```

The root `.env` supplies `GEMINI_API_KEY`; `SCHEMA_RELATION_MODEL` or `--model`
overrides the default `gemini-2.5-flash`. Batches contain at most ten unique pairs.
SQLite caching in `.cache/relations.sqlite3` keys judgments by model, prompt/schema
hash, bases, qualifier lists, and context. Canonical side ordering makes reversed
requests share a judgment. This enforces consistent ordering, rather than proving
the model itself is invariant to order. `--fresh` bypasses cached judgments.
Results include source, generation timestamp, model and prompt provenance.
Malformed responses fail explicitly and never fall back to accepting a pair.

## Checkpoint 9 — Evaluate relation classification

`relation_benchmark.json` contains 24 manually labeled cases: six positives,
ten hard negatives, and eight context-dependent cases. Wheel/tyre and raw/actual
are tested both with and without factual context. These provisional labels are
evaluation data, not universally valid domain assertions.

```bash
.venv/bin/python schema_matching/evaluate_relations.py
```

`evaluate_relations.py` reports accuracy separately by stratum and expected class,
a four-class confusion matrix, base-relation accuracy, hard negatives called
equivalent, and uncertain cases forced to equivalent. Every expected/actual result
and explanation is retained in `relation_benchmark_report.json`. Evaluation uses
fresh LLM calls by default; `--reuse-cache` reuses stored judgments. Only each
case's input fields go to the classifier; labels and strata stay in the evaluator.
No benchmark-based prompt tuning or training is performed.

## Checkpoint 10 — Final pair decision logic

`decide.py` returns `ACCEPT`, `REJECT`, or `UNCERTAIN` for each retrieved pair:

1. Reject when base similarity falls below the compatibility gate (default 0.90),
   skipping the relation API call for that pair.
2. Reject semantically incompatible bases.
3. Reject `CONTRASTING` or `RELATED_BUT_DIFFERENT` qualifiers, regardless of high
   lexical/embedding scores.
4. Retain `UNCERTAIN` when base or qualifier semantics are unresolved.
5. Retain `UNCERTAIN` for ambiguous decomposition or missing-versus-present
   qualifiers, even if the classifier returns equivalent.
6. Accept only `EQUIVALENT` qualifiers with compatible bases and either qualifier
   similarity >= 0.85 or lexical similarity >= 0.80. Otherwise retain uncertainty.

Thresholds are configurable via `--base-min`, `--qualifier-min`, and `--lexical-min`.
These remain provisional, model-dependent support thresholds. The old weighted
score is preserved as diagnostic evidence but is not consulted by the decision
policy. No high score overrides a semantic rejection or uncertainty.

Run on an existing score report:

```bash
.venv/bin/python schema_matching/decide.py --scores schema_matching/equivalence_report.json
```

Or run all implemented checkpoints from a CSV in one command:

```bash
.venv/bin/python schema_matching/decide.py --csv attributes.csv --column Attribute
```

Positional column names also work. Omit `--column` to read CSV headers. Optional
`--context` supplies a shared factual schema description. `--relation-model`
selects the classifier. Output defaults to `decision_report.json` and includes
each decision rule, semantic explanations, existing signals, and provenance.
Retrieval/embedding CLI options apply only to new input, not saved score reports.

Verify the policy against the existing Checkpoint 7 benchmark scores:

```bash
.venv/bin/python schema_matching/evaluate_decisions.py
```

`evaluate_decisions.py` records false accepts, accepted-pair precision/recall,
uncertainty, nonretrieved pairs, decision coverage, and accuracy on determinate
decisions in `decision_benchmark_report.json`. Abstentions are not counted as
correct decisions; positive abstentions/misses reduce acceptance recall. This
uses saved decomposition/embedding results and is not a fresh end-to-end parser
benchmark. It reuses matching relation-cache entries unless `--fresh` is supplied.

Offline verification across all checkpoints:

```bash
.venv/bin/python -m unittest discover -s schema_matching -p 'test_*.py' -v
```

Expected: 29 passing tests, including rejection precedence, uncertainty retention,
support thresholds, symmetric cache keys, cache refresh, output validation, and
stratified/abstention-aware metrics. These check implementation behavior, not LLM
semantic accuracy. Uncertain pairs remain available for review; no automatic
stronger-model escalation, probability calibration, or clustering is implemented.

### Initial Checkpoint 8–10 verification results

The network retry succeeded. All 29 offline tests passed, and the saved-score
runtime command completed, producing `decision_report.json`.

Relation benchmark (`relation_benchmark_report.json`):
- Overall: 21/24 correct (87.5%).
- Positives: 6/6; hard negatives: 10/10; context-dependent cases: 5/8.
- No hard negative or expected-uncertain case was called equivalent.
- Base compatibility labels matched in 23/24 cases.
- The three qualifier-label disagreements were raw/actual without context,
  absent/input qualifiers, and normal/standard. The model chose
  `RELATED_BUT_DIFFERENT` where the benchmark expected `UNCERTAIN`.

Decision benchmark (`decision_benchmark_report.json`, using the earlier saved
scores and their name-only context):
- Five correctly accepted positives, zero false accepts: acceptance precision
  100%, acceptance recall 5/7 (71.4%).
- One uncertain pair and one nonretrieved pair out of 15 labels.
- Determinate decision coverage 13/15 (86.7%); accuracy among those decisions
  12/13 (92.3%). These metrics explicitly exclude abstentions from accuracy.
- Input/incoming voltage is accepted; input/output voltage is rejected despite
  its higher old weighted score.
- Wheel/tyre position remains uncertain without context. Front-left wheel/tyre
  position was rejected as distinct components, missing a positive label from
  the prior benchmark. Both results remain visible rather than changing labels.

These are small development benchmarks, not independent validation of general
accuracy. Context dependence remains unresolved in some cases. No labels,
thresholds, or prompt instructions were changed after seeing these results.

## Checkpoint 11 — Confidence calibration

`calibrate.py` fits an empirical reliability profile from a decision report and
aligned manual pair labels. It groups observations by decision, decision rule,
and raw-score bin: [0, .8), [.8, .9), [.9, .95), [.95, 1]. Each bin records sample
count, correct decisions, empirical correctness, and a two-sided 95% Wilson
interval. This replaces treating the raw weighted score as a probability.

Defaults require at least 20 observations in the bin AND a correctness interval
lower bound of at least 0.90 before assigning a high-confidence band. The bound
is stricter than sample count alone (even 20/20 correct is insufficient). The
profile reports its assumptions and provenance. The interval describes bin-level
accuracy under sampling assumptions, not a guaranteed individual probability.
Correlated labels, model drift and domain shift can invalidate that interpretation;
independent representative calibration data and held-out evaluation remain needed.

```bash
.venv/bin/python schema_matching/calibrate.py fit
```

Default fitting inputs are the earlier `decision_benchmark_report.json` and
`equivalence_benchmark.json`; the result is `calibration_profile.json`. To fit
more representative data, pass `--decisions` and `--labels` with aligned columns
and pair IDs in the existing benchmark format. Only labeled, retrieved,
determinate decisions enter bins; uncertainty and retrieval misses are counted
separately. Duplicate labels are rejected. This calibrates outputs, not the LLM.

```bash
.venv/bin/python schema_matching/calibrate.py apply --decisions schema_matching/decision_report.json
```

This writes `confidence_report.json` with `HIGH_CONFIDENCE_MATCH`,
`HIGH_CONFIDENCE_NON_MATCH`, or `UNCERTAIN` per pair, plus supporting bin evidence
and the reason for abstention. ACCEPT/REJECT decisions gain their respective high
band only when the evidence gate passes. Semantic/decomposition/scope ambiguity
remains uncertain. Missing/insufficient calibration evidence also remains uncertain.
Profile compatibility checks classifier/prompt, policy, thresholds, context,
embedding settings, weights, and recorded parser/retrieval settings. Incompatible
profiles cause abstention rather than silent reuse. No automatic rule-only
fallback labels a pair as high confidence.

## Checkpoint 12 — Cluster confirmed matches

`cluster_confirmed.py` performs deterministic greedy complete-link merging:
every cross-group pair must be `HIGH_CONFIDENCE_MATCH` before merging two groups.
Missing, uncertain, or non-match edges block merging. It preserves original
column IDs (including repeated names) and emits singleton groups when evidence
is absent. It records blocked merges and a specific blocking pair. It does not
use connected components or infer missing edges from transitivity.

```bash
.venv/bin/python schema_matching/cluster_confirmed.py
```

This consumes `decision_report.json` and `calibration_profile.json`, applies bands,
clusters, audits, and writes `cluster_report.json` in one command. Override
`--decisions`, `--profile`, or `--output` as needed. An absent profile conservatively
produces uncertain pairs; an explicit invalid profile fails. Runtime does not read
benchmark labels or refit calibration.

To run the complete implemented pipeline from a CSV in one command:

```bash
.venv/bin/python schema_matching/cluster_confirmed.py --csv attributes.csv --column Attribute
```

Omit `--column` for CSV headers. Optional `--context` supplies a factual schema
description. New CSV input uses the root `.env` and existing default model/retrieval
settings; it requires Gemini for uncached model operations. Saved-decision grouping
and all calibration/audit commands are local and require no API calls.

Greedy grouping is reproducible for fixed IDs but can depend on their ordering
when multiple incompatible clique partitions exist. It does not claim the maximum
possible clique partition. Cross-group checks and all-pairs auditing can become
expensive for large groups; this is a conservative baseline, not an ANN or
large-scale graph optimization implementation.

## Checkpoint 13 — Validate cluster consistency

Every generated group is checked exhaustively. Each internal pair is reported as
EQUIVALENT, CONFLICTING, UNCERTAIN, or MISSING according to confidence evidence.
Every non-equivalent pair flags a proposed cluster. Each problematic closing edge
also gets an A–B–C chaining witness when both adjacent edges are confirmed matches.
Singletons are labeled SINGLETON, not claimed semantic equivalence discoveries.

Audit externally proposed clusters (JSON list of objects with `member_ids`, or an
object containing a `clusters` list):

```bash
.venv/bin/python schema_matching/cluster_confirmed.py --validate-clusters proposed_clusters.json --output schema_matching/cluster_audit.json
```

The command writes all diagnostics and exits with code 1 for flagged clusters.
Duplicate/overlapping members or invalid IDs are rejected. Unassigned column IDs
are reported separately for partial external proposals. Generated output always
assigns every column exactly once.

### Verification and current results

```bash
.venv/bin/python -m unittest discover -s schema_matching -p 'test_*.py' -v
.venv/bin/python schema_matching/evaluate_clustering.py
```

All 36 offline tests passed. All five fixtures in `clustering_benchmark.json`
passed: a complete clique, a conflicting chain, a missing closing edge, an
uncertain closing edge, and singletons. `clustering_benchmark_report.json` shows
both generated groups and intentionally proposed inconsistent clusters, with
pair evidence and chaining witnesses. These use synthetic bands solely to verify
mechanics; they are not evidence of semantic model quality or calibrated accuracy.

The real 15-label development data produced **zero eligible high-confidence
bins**. Applying that profile to the current saved report produced six uncertain
pair bands and **12 singleton groups**, with no consistency conflicts. This is
insufficient calibration evidence, not a conclusion that every attribute differs.
No sample counts were invented and no thresholds lowered to manufacture groups.
The saved-decision CLI, calibration fit/apply, and synthetic evaluator were run;
the new CSV orchestration path reuses earlier tested stages but was not rerun live.

Implementation for Checkpoints 11–13 is complete. Sufficient representative labels
and independent validation of confidence bands remain outstanding before the
current real-data run can form high-confidence multi-column groups.
