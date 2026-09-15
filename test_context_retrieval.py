import unittest
import numpy as np

from schema_context import profile_values, value_evidence, metadata
from retrieval import neighbors
from relations import canonical_request
from run import prepare_records
from error_taxonomy import categorize
from decide import decide_report


class ContextRetrievalTests(unittest.TestCase):
    def test_value_summaries_and_no_raw_samples(self):
        a = profile_values([1,2,3,None])
        self.assertEqual(a["numeric"]["min"],1)
        self.assertEqual(a["missing_fraction"],.25)
        self.assertEqual(profile_values([True,False])["observed_type"],"boolean")
        self.assertEqual(profile_values(["2026-01-01"])["observed_type"],"iso_datetime")
        e = value_evidence(dict(value_profile=a), dict(value_profile=profile_values([10,20])))
        self.assertFalse(e["ranges_overlap"])
        self.assertNotIn("category_hashes",e["left"])
        self.assertFalse(e["requires_review"])

    def test_categories_units_and_type_review(self):
        a = dict(schema={"units":"V"},value_profile=profile_values(["A","B"]))
        b = dict(schema={"units":"mV"},value_profile=profile_values(["B","C"]))
        e=value_evidence(a,b)
        self.assertAlmostEqual(e["categorical_jaccard"],1/3)
        self.assertTrue(e["requires_review"])
        self.assertFalse(value_evidence({}, {})["available"])

    def test_schema_survives_and_gold_decomposition_is_optional(self):
        class Parser:
            def extract(self,names,contexts):
                self.contexts=contexts
                return [dict(original=n,base="position",qualifiers=["wheel"],ambiguous=False,reason="test") for n in names]
        parser=Parser()
        items=prepare_records([dict(name="wheel_position",table="vehicle",units="slot",values=["FL","FR"])],parser)
        self.assertEqual(items[0]["schema"]["table"],"vehicle")
        self.assertIn("value_profile",items[0])
        self.assertNotIn("values",parser.contexts[0])

    def test_enriched_relation_cache_is_symmetric(self):
        a=dict(base_a="voltage",base_b="voltage",qualifiers_a=["input"],qualifiers_b=["output"],schema_a={"units":"V"},schema_b={"units":"mV"})
        b=dict(a,qualifiers_a=a["qualifiers_b"],qualifiers_b=a["qualifiers_a"],schema_a=a["schema_b"],schema_b=a["schema_a"])
        self.assertEqual(canonical_request(a),canonical_request(b))

    def test_hnsw_degree_bound_and_no_self_pairs(self):
        x=np.random.default_rng(7).normal(size=(600,32))
        edges,stats=neighbors(x,k=10,threshold=-1,method="hnsw")
        self.assertLessEqual(len(edges),3000)
        self.assertLessEqual(stats["max_final_degree"],10)
        self.assertTrue(all(i<j for i,j,_ in edges))
        duplicate=np.ones((40,16))
        edges,stats=neighbors(duplicate,k=3,threshold=.9)
        self.assertLessEqual(len(edges),60)

    def test_taxonomy_distinguishes_retrieval_and_calibration(self):
        c=dict(columns=[dict(base="rate",qualifiers=["heart"])]*2,expected_relation="EQUIVALENT",phenomenon="abbreviation")
        actual=c["columns"]
        self.assertEqual(categorize(c,actual,None,False),["retrieval_miss"])
        p=dict(relation={"relation":"EQUIVALENT"},decision="ACCEPT")
        self.assertEqual(categorize(c,actual,p,True),["insufficient_calibration"])

    def test_value_evidence_cannot_override_contrast_and_unit_review(self):
        class Classifier:
            settings={"model":"test"}
            def __init__(self,label): self.label=label
            def classify(self,requests,fresh=False):
                self.requests=requests
                return [dict(relation=self.label,base_relation="COMPATIBLE",reason="test",base_reason="test") for r in requests]
        columns=[dict(id=i,original=n,base="voltage",qualifiers=[q],ambiguous=False,reason="test",
                      schema={"units":u},value_profile=profile_values([1,2,3]))
                 for i,(n,q,u) in enumerate([("input_voltage","input","V"),("incoming_voltage","incoming","mV")])]
        scored=dict(columns=columns,scored_candidates=[dict(left_id=0,right_id=1,signals=dict(base=1,qualifier=1,lexical=1))])
        self.assertEqual(decide_report(scored,Classifier("EQUIVALENT"))["decisions"][0]["decision"],"UNCERTAIN")
        self.assertEqual(decide_report(scored,Classifier("CONTRASTING"))["decisions"][0]["decision"],"REJECT")

    def test_extreme_numeric_profiles_stay_finite(self):
        a=dict(value_profile=profile_values([-1e308,1e308]))
        b=dict(value_profile=profile_values([0,1]))
        self.assertTrue(np.isfinite(value_evidence(a,b)["normalized_quantile_distance"]))


if __name__=="__main__":
    unittest.main()
