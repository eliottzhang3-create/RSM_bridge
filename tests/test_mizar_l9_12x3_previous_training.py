"""Current complete-AudioMCQ policy rejects old mixed datasets."""
import unittest
from test_mizar_l9_12x3_strong_reason import recipe
class DatasetPolicyTests(unittest.TestCase):
    def test_reject_reason_or_incomplete_sources(self):
        for rows,sources in ((651296,{'reasonaqa_oneaudio':325648,'AudioCaps':325648}),(720523,recipe.SOURCE_COUNTS)):
            with self.assertRaises(ValueError):recipe.validate_data_counts(rows,sources)
    def test_fixed_full_audiomcq_inventory(self):
        self.assertEqual(sum(recipe.SOURCE_COUNTS.values()),720524)
        self.assertNotIn('reasonaqa_oneaudio',recipe.SOURCE_COUNTS)
        self.assertEqual(recipe.validate_data_counts(720524,recipe.SOURCE_COUNTS),720524)
if __name__=='__main__':unittest.main()
