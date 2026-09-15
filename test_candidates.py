import tempfile
from pathlib import Path
import unittest

import numpy as np

from candidates import generate_candidates, retrieve_bases, read_csv_names
from embeddings import BaseEmbedder, unit_vectors
from evaluate_candidates import retrieval_metrics


class FakeEmbedder:
    settings = {"backend": "synthetic_test_only"}
    cache_hits = 0
    embedded_count = 0

    def encode(self, bases):
        self.received = bases
        vectors = {"temp": [1, 0, 0], "temperature": [0.99, 0.1, 0],
                   "voltage": [0, 1, 0], "manufacturer": [0, 0, 1]}
        return np.array([vectors[b] for b in bases])


def column(name, base, qualifiers):
    return dict(original=name, base=base, qualifiers=qualifiers, ambiguous=False, reason="test")


class CandidateTests(unittest.TestCase):
    def test_semantic_and_exact_candidates_without_qualifier_filtering(self):
        items = [column("input_voltage", "voltage", ["input"]),
                 column("output_voltage", "voltage", ["output"]),
                 column("ambient_temp", "temp", ["ambient"]),
                 column("ambient_temperature", "temperature", ["ambient"]),
                 column("manufacturer", "manufacturer", [])]
        embedder = FakeEmbedder()
        report = generate_candidates(items, embedder)
        self.assertEqual({(p["left_id"], p["right_id"]) for p in report["candidates"]},
                         {(0, 1), (2, 3)})
        self.assertEqual(len(embedder.received), 4)
        self.assertEqual(report["all_possible_pairs"], 10)
        self.assertEqual(report["pair_reduction"], 0.8)

    def test_duplicate_columns_keep_ids_and_pair_limit_fails_explicitly(self):
        items = [column("input_voltage", "voltage", ["input"])] * 3
        report = generate_candidates(items, FakeEmbedder(), top_k=1)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["search"]["max_final_degree"], 1)
        with self.assertRaisesRegex(ValueError, "exceed"):
            generate_candidates(items, FakeEmbedder(), max_pairs=2)

    def test_empty_singleton_and_threshold(self):
        self.assertEqual(generate_candidates([], FakeEmbedder())["candidate_count"], 0)
        self.assertEqual(generate_candidates([column("temp", "temp", [])], FakeEmbedder())
                         ["candidate_count"], 0)
        self.assertEqual(retrieve_bases(["a", "b"], [[1, 0], [0, 1]], 0.9), [])
        with self.assertRaises(ValueError):
            retrieve_bases(["a"], [[1]], top_k=0)

    def test_blocked_search_matches_full_reference_and_has_no_self_pairs(self):
        vectors = unit_vectors(np.random.default_rng(42).normal(size=(17, 7)), 17)
        bases = [str(i) for i in range(17)]
        expected = set()
        scores = vectors @ vectors.T
        for i, row in enumerate(scores):
            eligible = [j for j in range(17) if i != j and row[j] >= 0.1]
            for j in sorted(eligible, key=lambda j: (-row[j], j))[:3]:
                expected.add(tuple(sorted((i, j))))
        for block_size in [1, 4, 256]:
            found = retrieve_bases(bases, vectors, 0.1, 3, block_size)
            self.assertEqual({(int(p["left_base"]), int(p["right_base"])) for p in found}, expected)

    def test_invalid_vectors(self):
        for vectors in [[[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [1, 2]]:
            with self.assertRaises(ValueError):
                unit_vectors(vectors, 1)

    def test_cache_reuses_vectors_and_separates_models(self):
        class Stub(BaseEmbedder):
            def _encode(self, texts):
                return [[1, 2] for _ in texts]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.sqlite3"
            first = Stub("gemini", "test-model-a", cache)
            first.encode(["temp", "temp", "temperature"])
            self.assertEqual(first.embedded_count, 2)
            second = Stub("gemini", "test-model-a", cache)
            second.encode(["temp", "temperature"])
            self.assertEqual(second.cache_hits, 2)
            self.assertEqual(second.embedded_count, 0)
            third = Stub("gemini", "test-model-b", cache)
            third.encode(["temp"])
            self.assertEqual(third.embedded_count, 1)

    def test_csv_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "names.csv"
            path.write_text("Attribute,other\ninput_voltage,x\noutput_voltage,y\n")
            self.assertEqual(read_csv_names(path), ["Attribute", "other"])
            self.assertEqual(read_csv_names(path, "Attribute"), ["input_voltage", "output_voltage"])
        rows = [{"candidate_group": "a"}, {"candidate_group": "a"}, {"candidate_group": "b"}]
        metrics = retrieval_metrics(rows, [{"left_id": 0, "right_id": 2}])
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["recall"], 0)


if __name__ == "__main__":
    unittest.main()
