import tempfile
from pathlib import Path
import unittest

from relations import RelationClassifier, canonical_request, validate_relations
from evaluate_relations import relation_metrics
from decide import decide_pair, decide_report
from evaluate_decisions import decision_metrics


def request(a="input", b="incoming", context=""):
    return dict(base_a="voltage", base_b="voltage", qualifiers_a=[a], qualifiers_b=[b], context=context)


def relation(label="EQUIVALENT", base="COMPATIBLE"):
    return dict(relation=label, base_relation=base, reason="test reason", base_reason="test base reason")


class StubClassifier(RelationClassifier):
    calls = 0

    def _predict(self, batch):
        self.calls += 1
        return [relation() for _ in batch]


class RelationTests(unittest.TestCase):
    def test_canonical_symmetric_order_and_context(self):
        a, b = request(), request("incoming", "input")
        self.assertEqual(canonical_request(a), canonical_request(b))
        self.assertNotEqual(canonical_request(a), canonical_request(request(context="specific schema")))
        with self.assertRaises(ValueError):
            canonical_request(dict(a, qualifiers_a="input"))

    def test_validation_missing_duplicate_invalid_and_reordered_ids(self):
        first, second = dict(id=0, **relation()), dict(id=1, **relation("CONTRASTING"))
        self.assertEqual(validate_relations({"items": [second, first]}, 2)[0]["relation"], "EQUIVALENT")
        for items in [[], [first, first], [dict(first, id=True)], [dict(first, relation="SIMILAR")],
                      [dict(first, reason="")], [dict(first, base_relation="SIMILAR")]]:
            with self.subTest(items=items), self.assertRaises(ValueError):
                validate_relations({"items": items}, 1)

    def test_cache_dedup_context_model_and_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relations.sqlite3"
            classifier = StubClassifier("test-model", path)
            results = classifier.classify([request(), request("incoming", "input")])
            self.assertEqual(len(results), 2)
            self.assertEqual(classifier.calls, 1)
            self.assertEqual(classifier.classify([request()])[0]["source"], "cache")
            classifier.classify([request(context="new context")])
            self.assertEqual(classifier.calls, 2)
            classifier.classify([request()], fresh=True)
            self.assertEqual(classifier.calls, 3)
            other = StubClassifier("other-model", path)
            self.assertEqual(other.classify([request()])[0]["source"], "model")

    def test_metrics_keep_uncertainty_and_strata_separate(self):
        cases = [dict(expected="EQUIVALENT", expected_base="COMPATIBLE", stratum="positives"),
                 dict(expected="CONTRASTING", expected_base="COMPATIBLE", stratum="hard_negatives"),
                 dict(expected="UNCERTAIN", expected_base="UNCERTAIN", stratum="context_dependent")]
        metrics = relation_metrics(cases, [relation(), relation(), relation("UNCERTAIN", "UNCERTAIN")])
        self.assertEqual(metrics["overall"]["accuracy"], 2 / 3)
        self.assertEqual(metrics["by_stratum"]["hard_negatives"]["accuracy"], 0)
        self.assertEqual(metrics["hard_negatives_called_equivalent"], 1)


class DecisionTests(unittest.TestCase):
    def test_vetoes_override_perfect_scores(self):
        signals = dict(base=1, qualifier=1, lexical=1)
        for rel in [relation("CONTRASTING"), relation("RELATED_BUT_DIFFERENT"), relation(base="INCOMPATIBLE")]:
            self.assertEqual(decide_pair(signals, rel)["decision"], "REJECT")
        self.assertEqual(decide_pair(dict(signals, base=0.89), relation())["decision"], "REJECT")

    def test_uncertainty_never_overridden_by_high_scores(self):
        signals = dict(base=1, qualifier=1, lexical=1)
        for rel in [relation("UNCERTAIN"), relation(base="UNCERTAIN"), None]:
            self.assertEqual(decide_pair(signals, rel)["decision"], "UNCERTAIN")
        self.assertEqual(decide_pair(signals, relation(), ambiguous=True)["decision"], "UNCERTAIN")
        self.assertEqual(decide_pair(signals, relation(), presence_mismatch=True)["decision"], "UNCERTAIN")

    def test_equivalent_needs_evidence_and_threshold_boundaries(self):
        signals = dict(base=0.9, qualifier=0.85, lexical=0)
        self.assertEqual(decide_pair(signals, relation())["decision"], "ACCEPT")
        self.assertEqual(decide_pair(dict(signals, qualifier=0.84), relation())["decision"], "UNCERTAIN")
        self.assertEqual(decide_pair(dict(signals, qualifier=0.1, lexical=0.8), relation())["decision"], "ACCEPT")
        with self.assertRaises(ValueError):
            decide_pair(dict(signals, qualifier=float("nan")), relation())

    def test_pipeline_skips_incompatible_pairs_and_preserves_ids(self):
        class NoCall:
            settings = {"model": "test"}

            def classify(self, requests, fresh=False):
                if requests:
                    raise AssertionError("low base similarity must skip LLM")
                return []
        columns = [dict(id=i, original=name, base=name, qualifiers=[], ambiguous=False, reason="test")
                   for i, name in enumerate(["voltage", "manufacturer"])]
        scored = dict(columns=columns, scored_candidates=[dict(left_id=0, right_id=1,
                      signals=dict(base=0.5, qualifier=1, lexical=1), equivalence_score=1)])
        output = decide_report(scored, NoCall())
        self.assertEqual(output["decisions"][0]["decision"], "REJECT")
        self.assertIsNone(output["decisions"][0]["relation"])
        self.assertEqual(output["classified_pairs"], 0)

    def test_decision_metrics_count_abstentions_and_missing_pairs(self):
        benchmark = dict(columns=[dict(name=n) for n in ["a", "b", "c"]], pairs=[
            dict(left_id=0, right_id=1, equivalent=True), dict(left_id=0, right_id=2, equivalent=False),
            dict(left_id=1, right_id=2, equivalent=True)])
        report = dict(columns=[dict(original=n) for n in ["a", "b", "c"]], decisions=[
            dict(left_id=0, right_id=1, decision="UNCERTAIN"), dict(left_id=0, right_id=2, decision="REJECT")])
        metrics = decision_metrics(benchmark, report)
        self.assertEqual(metrics["accept_recall"], 0)
        self.assertEqual(metrics["uncertain"], 1)
        self.assertEqual(metrics["not_retrieved"], 1)
        self.assertEqual(metrics["decision_coverage"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
