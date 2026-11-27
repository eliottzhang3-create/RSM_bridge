"""Strong-only dynamic quotas remain bounded by original S1 metadata."""
import unittest
from test_mizar_l9_12x3_strong_reason import recipe
class DatasetPolicyTests(unittest.TestCase):
    def test_reject_reason_or_overbound_and_wrong_views(self):
        for rows,sources,views in ((6,{'reasonaqa_oneaudio':3,'AudioCaps':3},None),(325649,recipe.SOURCE_COUNTS,None),(3,{'AudioCaps':3},{'reasonaqa':3})):
            with self.assertRaises(ValueError):recipe.validate_data_counts(rows,sources,views)
    def test_retained_quota_measured_not_assumed_original(self):
        self.assertEqual(sum(recipe.SOURCE_COUNTS.values()),325648)
        self.assertEqual(recipe.validate_data_counts(3,{'AudioCaps':3},{'audiomcq_mcq':2,'audiomcq_cot':1}),3)
if __name__=='__main__':unittest.main()
