# Original SmolLM2-135M audio baseline

This route is an isolated comparison line for the current ReasonAQA audio
training contract.  It reuses the existing 32 kHz/10 second waveform path,
Mellow c2l adapter, 768-to-576 projection bridge, 129-token-per-audio prefix,
260-token multimodal prefix, unified dynamic padding, and answer-only labels.

The text backbone is loaded directly with `AutoModelForCausalLM` from the
local original SmolLM2 directory and is validated as a standard
`LlamaForCausalLM` with 30 independent decoder layers and hidden size 576.
HTSAT is frozen; c2l, the audio bridge, and all original text-model
parameters are trainable.  Checkpoints use `text_model/`, `tokenizer/`,
`audio_bridge.pt`, `training_state.pt`, `audio_smollm2_config.json`, and an
atomic `checkpoint_complete.json` marker.

GPU execution goes through these repository wrappers:

```text
run_audio_smollm2_135m_mellow_smoke20_5090.sh
run_audio_smollm2_135m_mellow_resume2_5090.sh
run_audio_smollm2_135m_mellow_formal_5090.sh
run_audio_smollm2_135m_mellow_checkpoint_audit_5090.sh
```

The smoke run executes only through step 20 while using a shared short-run
scheduler of 22 steps, and saves steps 10 and 20.  The resume wrapper restores
step 20 and continues to step 22 in a separate output directory; its report
also proves that representative text, bridge, and c2l parameters changed.
The first formal invocation starts from the original
`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2` directory.  A later formal
invocation may resume only from a checkpoint whose gate is `FORMAL`; smoke
checkpoints are never accepted as formal starting points.
Every resume, including FORMAL resume, must use an output directory separate
from both the source checkpoint and its parent directory.

For example, keep the smoke and resume artifacts isolated:

```bash
SMOKE_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/smoke20_20260911
RESUME_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/resume2_from20_20260911

bash code/RSmol/run_audio_smollm2_135m_mellow_smoke20_5090.sh \
  --output-dir "$SMOKE_OUT"
bash code/RSmol/run_audio_smollm2_135m_mellow_resume2_5090.sh \
  --resume-from "$SMOKE_OUT/checkpoint-000020" \
  --output-dir "$RESUME_OUT"
bash code/RSmol/run_audio_smollm2_135m_mellow_checkpoint_audit_5090.sh \
  --checkpoint "$RESUME_OUT/checkpoint-000022" \
  --parent-checkpoint "$SMOKE_OUT/checkpoint-000020" \
  --expected-parent-step 20 \
  --model-path /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  --manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --val-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_val.jsonl
```

The formal run uses its own output directory and the original model path:

```bash
FORMAL_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/formal
FORMAL_RESUME_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/formal_resume_from500

bash code/RSmol/run_audio_smollm2_135m_mellow_formal_5090.sh \
  --model-path /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  --output-dir "$FORMAL_OUT"

# After an interruption, resume only from a FORMAL checkpoint (for example,
# checkpoint-000500), and always use a separate continuation directory.
bash code/RSmol/run_audio_smollm2_135m_mellow_formal_5090.sh \
  --model-path /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  --resume-from "$FORMAL_OUT/checkpoint-000500" \
  --output-dir "$FORMAL_RESUME_OUT"
```

Audit a formal checkpoint by supplying its actual global step; for a
continuation, also supply the parent checkpoint so the immutable contract and
gate are checked across the resume edge:

```bash
bash code/RSmol/run_audio_smollm2_135m_mellow_checkpoint_audit_5090.sh \
  --checkpoint "$FORMAL_OUT/checkpoint-000500" \
  --expected-gate FORMAL --expected-step 500 \
  --model-path /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  --manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --val-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_val.jsonl
```

The trainer records and validates `gate`/`run_kind` in every checkpoint and
training state.  A STAGE7 smoke or resume checkpoint is rejected as a FORMAL
resume source; a FORMAL continuation must retain the canonical three-epoch
schedule, manifests, provenance, optimizer, batch, and checkpoint cadence.
