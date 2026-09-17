import unittest

from calibrate import semantic_bypass_report


class SemanticBypassTests(unittest.TestCase):
    def test_maps_only_final_semantic_decisions(self):
        report = {
            "columns": [{"id": 0}, {"id": 1}, {"id": 2}],
            "decisions": [
                {"left_id": 0, "right_id": 1, "decision": "ACCEPT",
                 "decision_rule": "test", "equivalence_score": 0.9},
                {"left_id": 0, "right_id": 2, "decision": "REJECT",
                 "decision_rule": "test", "equivalence_score": 0.1},
                {"left_id": 1, "right_id": 2, "decision": "UNCERTAIN",
                 "decision_rule": "test", "equivalence_score": 0.5},
            ],
        }
        result = semantic_bypass_report(report)
        self.assertTrue(result["bypassed"])
        self.assertEqual([p["confidence_band"] for p in result["pairs"]], [
            "HIGH_CONFIDENCE_MATCH", "HIGH_CONFIDENCE_NON_MATCH", "UNCERTAIN"])
        self.assertTrue(all(p["calibration_status"] == "BYPASSED" for p in result["pairs"]))


if __name__ == "__main__":
    unittest.main()
