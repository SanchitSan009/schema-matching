import tempfile
import unittest
from pathlib import Path

from decompose import Parser
from relations import RelationClassifier


class FakeParser(Parser):
    def __init__(self, cache, responses):
        super().__init__(model="test-model", cache=cache)
        self.responses = iter(responses)
        self.calls = []

    def _extract_batch(self, batch, contexts):
        self.calls.append(list(batch))
        response = next(self.responses)
        return [response.get(name) for name in batch]


class FakeClassifier(RelationClassifier):
    def __init__(self, cache, responses):
        super().__init__(model="test-model", cache=cache)
        self.responses = iter(responses)
        self.calls = []

    def _predict(self, batch):
        self.calls.append(list(batch))
        response = next(self.responses)
        return response[:len(batch)]


def decomposition(base, qualifier):
    return {"base": base, "qualifiers": [qualifier], "ambiguous": False, "reason": "test"}


class ApiBatchingTests(unittest.TestCase):
    def test_decomposition_deduplicates_and_retries_only_missing_items(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "decompose.sqlite3"
            parser = FakeParser(cache, [
                {"input voltage": decomposition("voltage", "input")},
                {"output voltage": decomposition("voltage", "output")},
            ])
            result = parser.extract(["input_voltage", "output_voltage", "input-voltage"])
            self.assertEqual(parser.calls, [
                ["input voltage", "output voltage"], ["output voltage"]])
            self.assertEqual([row["base"] for row in result], ["voltage"] * 3)

            cached = FakeParser(cache, [])
            cached.extract(["output_voltage", "input_voltage"])
            self.assertEqual(cached.calls, [])
            self.assertEqual(cached.cache_hits, 2)

    def test_relations_are_symmetric_cached_and_retry_only_missing_items(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "relations.sqlite3"
            equivalent = {"relation": "EQUIVALENT", "base_relation": "COMPATIBLE"}
            contrasting = {"relation": "CONTRASTING", "base_relation": "COMPATIBLE"}
            requests = [
                dict(base_a="voltage", qualifiers_a=["input"],
                     base_b="voltage", qualifiers_b=["incoming"]),
                dict(base_a="voltage", qualifiers_a=["input"],
                     base_b="voltage", qualifiers_b=["output"]),
            ]
            classifier = FakeClassifier(cache, [[equivalent, None], [contrasting]])
            result = classifier.classify(requests)
            self.assertEqual([len(call) for call in classifier.calls], [2, 1])
            self.assertEqual([row["relation"] for row in result], ["EQUIVALENT", "CONTRASTING"])

            reversed_request = dict(base_a="voltage", qualifiers_a=["incoming"],
                                    base_b="voltage", qualifiers_b=["input"])
            cached = FakeClassifier(cache, [])
            self.assertEqual(cached.classify([reversed_request])[0]["source"], "cache")
            self.assertEqual(cached.calls, [])

    def test_exact_relation_is_local(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = FakeClassifier(Path(directory) / "relations.sqlite3", [])
            result = classifier.classify([dict(base_a="part no", qualifiers_a=[],
                                               base_b="part_no", qualifiers_b=[])])
            self.assertEqual(result[0]["source"], "local_exact")
            self.assertEqual(classifier.calls, [])


if __name__ == "__main__":
    unittest.main()
