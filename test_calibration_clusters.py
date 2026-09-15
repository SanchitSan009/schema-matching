import json
import unittest

from calibrate import HERE, confidence_report, fit_profile, signature, bucket, wilson
from cluster_confirmed import cluster_confirmed, validate_clusters
from evaluate_clustering import fixture_report


def decisions():
    return dict(classifier={"model": "test"}, decision_policy="test", support_thresholds={},
                columns=[dict(id=i, original=n, qualifiers=["input"], ambiguous=False) for i, n in enumerate(["a", "b"])],
                decisions=[dict(left_id=0, right_id=1, equivalence_score=0.96, decision="ACCEPT", decision_rule="test")])


class CalibrationTests(unittest.TestCase):
    def test_small_sample_abstains_even_when_perfect(self):
        report = decisions()
        labels = dict(columns=[dict(name=n) for n in ["a", "b"]], pairs=[dict(left_id=0, right_id=1, equivalent=True)])
        profile = fit_profile(report, labels)
        self.assertEqual(profile["high_confidence_bin_count"], 0)
        self.assertEqual(confidence_report(report, profile)["pairs"][0]["confidence_band"], "UNCERTAIN")
        self.assertLess(wilson(5, 5)[0], 0.9)

    def test_sufficient_evidence_bands_and_semantic_veto(self):
        report = decisions()
        labels = dict(columns=[dict(name=n) for n in ["a", "b"]], pairs=[dict(left_id=0, right_id=1, equivalent=True)])
        profile = fit_profile(report, labels)
        # Synthetic counts exercise the statistical gate; not model evidence.
        entry = profile["bins"][bucket(report["decisions"][0])]
        entry.update(count=100, correct=100)
        self.assertEqual(confidence_report(report, profile)["pairs"][0]["confidence_band"], "HIGH_CONFIDENCE_MATCH")
        report["columns"][0]["ambiguous"] = True
        self.assertEqual(confidence_report(report, profile)["pairs"][0]["confidence_band"], "UNCERTAIN")

    def test_nonmatch_and_profile_mismatch(self):
        report = decisions()
        report["decisions"][0]["decision"] = "REJECT"
        labels = dict(columns=[dict(name=n) for n in ["a", "b"]], pairs=[dict(left_id=0, right_id=1, equivalent=False)])
        profile = fit_profile(report, labels)
        profile["bins"][bucket(report["decisions"][0])].update(count=100, correct=100)
        self.assertEqual(confidence_report(report, profile)["pairs"][0]["confidence_band"], "HIGH_CONFIDENCE_NON_MATCH")
        report["classifier"] = {"model": "different"}
        self.assertEqual(confidence_report(report, profile)["pairs"][0]["confidence_band"], "UNCERTAIN")
        self.assertEqual(confidence_report(report)["pairs"][0]["confidence_band"], "UNCERTAIN")

    def test_duplicate_labels_and_invalid_scores_rejected(self):
        report = decisions()
        labels = dict(columns=[dict(name=n) for n in ["a", "b"]], pairs=[dict(left_id=0, right_id=1, equivalent=True)] * 2)
        with self.assertRaises(ValueError):
            fit_profile(report, labels)
        report["decisions"][0]["equivalence_score"] = float("nan")
        with self.assertRaises(ValueError):
            confidence_report(report)


class ClusterTests(unittest.TestCase):
    def test_all_synthetic_fixtures(self):
        cases = json.loads((HERE / "clustering_benchmark.json").read_text())["cases"]
        for case in cases:
            with self.subTest(case=case["name"]):
                report = fixture_report(case)
                clustered = cluster_confirmed(report)
                self.assertTrue(clustered["consistency_passed"])
                self.assertEqual(sorted(len(c["member_ids"]) for c in clustered["clusters"]), case["expected_group_sizes"])
                audit = validate_clusters(report, [dict(member_ids=list(range(case["size"])))])
                self.assertEqual(audit["passed"], case["proposed_cluster_passes"])
                if "chain" in case["name"] or "closing" in case["name"]:
                    self.assertTrue(audit["clusters"][0]["chaining_witnesses"])

    def test_missing_edges_and_overlapping_clusters(self):
        report = dict(columns=[dict(id=i, original="duplicate") for i in range(3)], pairs=[])
        self.assertEqual(cluster_confirmed(report)["cluster_count"], 3)
        with self.assertRaises(ValueError):
            validate_clusters(report, [dict(member_ids=[0, 1]), dict(member_ids=[1, 2])])
        self.assertEqual(cluster_confirmed(dict(columns=[], pairs=[]))["cluster_count"], 0)

    def test_deterministic_partition_preserves_every_member(self):
        report = fixture_report(dict(size=4, matches=[[0,1],[1,2],[2,3]], non_matches=[], uncertain=[]))
        first = cluster_confirmed(report)
        report["pairs"].reverse()
        second = cluster_confirmed(report)
        self.assertEqual(first, second)
        self.assertEqual(sorted(i for c in first["clusters"] for i in c["member_ids"]), list(range(4)))


if __name__ == "__main__":
    unittest.main()
