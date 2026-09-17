# Controlled retrieval experiments

The cross-domain development stress benchmark has 32 cases, including 16 labeled
equivalent pairs. Each domain is retrieved as its own eight-column collection.
The runs reuse the saved parser output in `cross_domain_report.json` and cached
Gemini `gemini-embedding-001` vectors. No parser or embedding model calls were made.
The baseline rerun matches the saved report's recall and per-domain candidate counts.

| Run | Threshold | Top-k | Positive recall | Candidates | Change from baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.90 | 15 | 15/16 (93.75%) | 27/224 | — |
| A: threshold only | 0.88 | 15 | 16/16 (100%) | 39/224 | +12 candidates (+44.4%) |
| B: top-k only | 0.90 | 25 | 15/16 (93.75%) | 27/224 | No change |

Run A recovers `case_26`. Its extra 12 candidates are manageable at this
benchmark size, so **0.88 is the preferred setting for the next validation run**.
Run B has no effect because each independent collection has only eight columns;
top-k=15 already exceeds its possible neighbor count. Keep top-k at 15.

The benchmark is development data, not a certified unseen-domain holdout. A
single recovered positive pair cannot establish that 0.88 generalizes; verify
recall and candidate volume on a larger independent collection before changing
the global default. Candidate counts measure pairs sent to downstream stages,
not final accepted matches.

Reproduce each run separately with `experiment_retrieval.py` and the settings in
the table. Detailed results are in `retrieval_experiment_baseline.json`,
`retrieval_experiment_threshold_088.json`, and `retrieval_experiment_topk_25.json`.
