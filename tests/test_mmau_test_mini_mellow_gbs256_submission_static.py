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
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_3090/formal_5epochs_maxlr_1e-3_20261009_234938223175399/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_oldmean_formal_20_20261009_213956781718666/model--epo-5.ckpt",
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_1e-3_20261009_151832898359917/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261009_074133259050800/model--epo-5.ckpt",
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_2e-3_20261009_234954643835601/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261009_233443558075953/model--epo-5.ckpt",
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_2e-4_20261010_210025620938234/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261010_154021644553389/model--epo-5.ckpt",
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_4e-4_20261010_210023713300705/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261010_132413418875990/model--epo-5.ckpt",
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_8e-4_20261010_210000727826895/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261010_150900245347077/model--epo-5.ckpt",
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
                r'^  "(/hpc_stor03/sjtu_home/jinwei\.zhang/outputs/RSmol/.+?/model--epo-5\.ckpt)"$',
                source,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(checkpoints, EXPECTED_CHECKPOINTS)
        self.assertIn('if bash "$SINGLE_SUBMIT" "$checkpoint" "" "" "$MODE"; then', source)
        self.assertIn('submitted=$submitted failed=$failed mode=$MODE', source)


if __name__ == "__main__":
    unittest.main()

