import unittest

from decompose import validate_response
from evaluate import evaluate


def item(idx=0, base="voltage", qualifiers=None):
    return dict(id=idx, base=base, qualifiers=["input"] if qualifiers is None else qualifiers,
                ambiguous=False, reason="test")


class DecompositionTests(unittest.TestCase):
    def test_reorders_ids_and_preserves_duplicate_inputs(self):
        result = validate_response({"items": [item(1), item(0)]},
                                   ["input voltage", "input voltage"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["base"], "voltage")

    def test_rejects_corrupt_responses(self):
        for items in [[], [item(), item()], [item(2)],
                      [item(base="current")], [item(qualifiers=[])],
                      [item(qualifiers=["input", "input"])]]:
            with self.subTest(items=items), self.assertRaises(ValueError):
                validate_response({"items": items}, ["input voltage"])

    def test_metrics_count_errors_and_ignore_qualifier_order(self):
        gold = [dict(name="front_left_position", base="position", qualifiers=["front", "left"]),
                dict(name="input_voltage", base="voltage", qualifiers=["input"])]
        predicted = [dict(original=gold[0]["name"], base="position",
                          qualifiers=["left", "front"], ambiguous=False),
                     dict(original=gold[1]["name"], base="input",
                          qualifiers=["voltage"], ambiguous=True)]
        report = evaluate(gold, predicted)
        self.assertEqual(report["exact_accuracy"], 0.5)
        self.assertEqual(report["base_accuracy"], 0.5)
        self.assertEqual(report["qualifier_accuracy"], 0.5)
        self.assertEqual(report["ambiguous_count"], 1)
        with self.assertRaises(ValueError):
            evaluate(gold, predicted[:1])


if __name__ == "__main__":
    unittest.main()
