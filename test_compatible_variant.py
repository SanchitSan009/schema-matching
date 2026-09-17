import unittest

from decide import decide_pair
from relations import validate_relations


class CompatibleVariantTests(unittest.TestCase):
    def test_relation_schema_accepts_compatible_variant(self):
        result = validate_relations({"items": [{
            "id": 0, "relation": "COMPATIBLE_VARIANT", "base_relation": "COMPATIBLE"
        }]}, 1)
        self.assertEqual(result[0]["relation"], "COMPATIBLE_VARIANT")

    def test_supported_variant_accepts_presence_mismatch(self):
        result = decide_pair(
            {"base": 1.0, "qualifier": 0.9, "lexical": 0.5},
            {"relation": "COMPATIBLE_VARIANT", "base_relation": "COMPATIBLE"},
            presence_mismatch=True,
        )
        self.assertEqual(result, {
            "decision": "UNCERTAIN", "decision_rule": "compatible_variant_requires_review"})

    def test_unsupported_variant_abstains(self):
        result = decide_pair(
            {"base": 1.0, "qualifier": 0.4, "lexical": 0.5},
            {"relation": "COMPATIBLE_VARIANT", "base_relation": "COMPATIBLE"},
        )
        self.assertEqual(result["decision"], "UNCERTAIN")

    def test_strict_equivalence_still_abstains_on_presence_mismatch(self):
        result = decide_pair(
            {"base": 1.0, "qualifier": 0.9, "lexical": 0.9},
            {"relation": "EQUIVALENT", "base_relation": "COMPATIBLE"},
            presence_mismatch=True,
        )
        self.assertEqual(result["decision"], "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
