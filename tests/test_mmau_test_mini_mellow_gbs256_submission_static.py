"""Static contracts for gbs256 MMAU submission entry points."""
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SINGLE_SUBMIT = RSMOL / "submit_mmau_test_mini_mellow_gbs256_3090.sh"
BATCH_SUBMIT = RSMOL / "submit_mmau_test_mini_mellow_gbs256_ablation_3090.sh"


EXPECTED_CHECKPOINTS = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/formal_10epochs_lr5_frozen_clip_maxlr_2e-5_20261018_181826682119766/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20_20261018_101831978638346/model--epo-10.ckpt",
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/formal_10epochs_lr5_frozen_clip_maxlr_3e-5_20261018_181649446900432/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20_20261018_101654725998277/model--epo-10.ckpt",
)


class GBS256SubmissionStaticTest(unittest.TestCase):
    def test_single_submitter_reports_preflight_and_vc_errors(self) -> None:
        source = SINGLE_SUBMIT.read_text(encoding="utf-8")
        self.assertIn('CHECKPOINT=$(readlink -f -- "$1") || die', source)
        self.assertIn('[[ -f "$CHECKPOINT" ]] || die "checkpoint not found:', source)
        self.assertIn('if VC_OUTPUT=$(vc submit', source)
        self.assertIn('[gbs256-submit] vc submit failed', source)
        self.assertIn('mellow-gbs256-mmau-${MODE}-3090-', source)

    def test_batch_submitter_contains_the_six_requested_checkpoints(self) -> None:
        source = BATCH_SUBMIT.read_text(encoding="utf-8")
        checkpoints = tuple(
            re.findall(
                r'^  "(/hpc_stor03/sjtu_home/jinwei\.zhang/outputs/RSmol/.+?/model--epo-10\.ckpt)"$',
                source,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(checkpoints, EXPECTED_CHECKPOINTS)
        self.assertIn('if bash "$SINGLE_SUBMIT" "$checkpoint" "" "" "$MODE"; then', source)
        self.assertIn('submitted=$submitted failed=$failed mode=$MODE', source)


if __name__ == "__main__":
    unittest.main()

