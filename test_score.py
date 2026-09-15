import hashlib
import math
import unittest

import numpy as np

from evaluate_scores import evaluate_scores
from score import CONTEXT_FORMAT, lexical_similarity, score_candidates, validate_weights


SETTINGS = dict(backend="test", model="synthetic", input_format="bare-base-v1", task="SEMANTIC_SIMILARITY")


class FakeEmbedder:
    settings = dict(SETTINGS, input_format=CONTEXT_FORMAT)
    cache_hits = 0
    embedded_count = 0

    def encode(self, texts):
        self.texts = texts
        self.embedded_count += len(texts)
        return np.array([list(hashlib.sha256(text.encode()).digest()) for text in texts], dtype=float)


def column(base, qualifiers):
    name = "_".join([*qualifiers, base])
    return dict(original=name, base=base, qualifiers=qualifiers, ambiguous=False, reason="test")


def report(left, right):
    return dict(embedding=SETTINGS, columns=[dict(id=0, **left), dict(id=1, **right)],
                candidates=[dict(left_id=0, right_id=1, base_similarity=0.93)])


class ScoreTests(unittest.TestCase):
    def test_shared_contexts_formula_and_swap_symmetry(self):
        left, right = column("temp", ["input"]), column("temperature", ["incoming"])
        embedder = FakeEmbedder()
        result = score_candidates(report(left, right), embedder, isolated_baseline=True)
        scored = result["scored_candidates"][0]
        self.assertEqual([c["base"] for c in scored["qualifier_contexts"]], ["temp", "temperature"])
        for context in scored["qualifier_contexts"]:
            self.assertTrue(context["left_text"].startswith("Attribute property: " + context["base"] + "\n"))
            self.assertTrue(context["right_text"].startswith("Attribute property: " + context["base"] + "\n"))
        self.assertAlmostEqual(scored["equivalence_score"],
            0.25 * scored["signals"]["base"] + 0.65 * scored["signals"]["qualifier"] + 0.10 * scored["signals"]["lexical"])
        reverse = score_candidates(report(right, left), FakeEmbedder())["scored_candidates"][0]
        self.assertAlmostEqual(reverse["equivalence_score"], scored["equivalence_score"])
        self.assertEqual(len(embedder.texts), len(set(embedder.texts)))

    def test_qualifier_order_empty_and_missing_scope(self):
        scored = score_candidates(report(column("position", ["front", "left"]),
                                          column("position", ["left", "front"])), FakeEmbedder())["scored_candidates"][0]
        self.assertAlmostEqual(scored["signals"]["qualifier"], 1.0, places=6)
        empty = column("voltage", [])
        scored = score_candidates(report(empty, empty), FakeEmbedder())["scored_candidates"][0]
        self.assertAlmostEqual(scored["signals"]["qualifier"], 1.0, places=6)
        scored = score_candidates(report(empty, column("voltage", ["input"])), FakeEmbedder())["scored_candidates"][0]
        self.assertTrue(scored["qualifier_presence_mismatch"])
        self.assertIn("[]", scored["qualifier_contexts"][0]["left_text"])

    def test_bad_weights_are_rejected(self):
        for weights in [(1, 1, 1), (-1, 1, 1), (math.nan, 0, 1), (math.inf, 0, 0), (1, 0)]:
            with self.assertRaises(ValueError):
                validate_weights(weights)

    def test_lexical_spelling_support_and_symmetry(self):
        self.assertGreater(lexical_similarity("colour", "color"), 0.9)
        self.assertGreater(lexical_similarity("temperature", "temprature"), 0.9)
        self.assertEqual(lexical_similarity("tide", "diet"), lexical_similarity("diet", "tide"))
        self.assertEqual(lexical_similarity("InputVoltage", "input_voltage"), 1)

    def test_no_candidates_does_not_call_embeddings(self):
        data = dict(embedding=SETTINGS, columns=[], candidates=[])
        embedder = FakeEmbedder()
        self.assertEqual(score_candidates(data, embedder)["scored_candidates"], [])
        self.assertFalse(hasattr(embedder, "texts"))

    def test_invalid_candidate_reports_rejected(self):
        c = column("voltage", ["input"])
        for similarity in [math.nan, math.inf, 2, True]:
            data = report(c, c)
            data["candidates"][0]["base_similarity"] = similarity
            with self.assertRaises(ValueError):
                score_candidates(data, FakeEmbedder())
        data = report(c, c)
        data["candidates"] *= 2
        with self.assertRaises(ValueError):
            score_candidates(data, FakeEmbedder())
        data = report(c, c)
        data["embedding"] = dict(SETTINGS, model="other")
        with self.assertRaises(ValueError):
            score_candidates(data, FakeEmbedder())

    def test_evaluation_reports_ranking_failure_and_retrieval_miss(self):
        benchmark = dict(columns=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
            pairs=[dict(left_id=0, right_id=1, equivalent=True),
                   dict(left_id=0, right_id=2, equivalent=False),
                   dict(left_id=1, right_id=2, equivalent=True)],
            rankings=[dict(name="test", positive=[0, 1], negative=[0, 2])])
        scored = dict(scored_candidates=[dict(left_id=0, right_id=1, equivalence_score=0.8,
                          qualifier_cosine=0.7, isolated_qualifier_cosine=0.9, signals={"base": 1, "lexical": 0.5}),
                      dict(left_id=0, right_id=2, equivalence_score=0.95,
                          qualifier_cosine=0.98, isolated_qualifier_cosine=0.6, signals={"base": 1, "lexical": 0.5})])
        metrics = evaluate_scores(benchmark, scored)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 2)
        self.assertEqual(metrics["positive_retrieval_recall"], 0.5)
        self.assertEqual(metrics["retained_pair_auc"]["contextual"], 0)
        self.assertEqual(metrics["retained_pair_auc"]["isolated"], 1)
        self.assertFalse(metrics["rankings"][0]["qualifier_cosine"]["correct_order"])


if __name__ == "__main__":
    unittest.main()
