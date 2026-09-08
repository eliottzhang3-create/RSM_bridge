# Isolated Mellow 5-10-5 audio audits

These entry points are independent of the legacy text, MeSH, and Parcae
variants. Remote code, data, and checkpoints are supplied at runtime; no
remote file is copied into this checkout.

Stage 0 is CPU-only and audits package versions, source visibility, JSON
splits, and checkpoint readability. It does not read waveforms:

```bash
python code/RSmol/scripts/audit_audio_stage0_5_10_5_mellow.py \
  --report-path /tmp/audio-stage0.json \
  --mellow-root /hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main \
  --htsat-root /hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main \
  --htsat-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt
```

Stage 1 builds deterministic JSONL manifests. An empty `filepath2` is always
represented by the same resolved `filepath1`; each such row has
`audio2_reused=true`, `audio2_source=filepath1_duplicate`, and
`is_duplicate=true`. No random sampling is used. Missing or ambiguous paths
fail unless `--allow-missing` is explicitly supplied:

```bash
python code/RSmol/scripts/prepare_reasonaqa_manifest_5_10_5_mellow.py \
  --reasonaqa-root /hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa \
  --audiocaps-root /hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2 \
  --clotho-audio-root /hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1 \
  --clotho-aqa-audio-root /hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio \
  --output-dir /tmp/reasonaqa-manifest
```

Clotho-AQA rows are resolved exclusively from the official
`clotho_aqa_audio/audio_files` tree.  They are not mapped to Clotho v2.1 by
basename normalization, and a missing AQA file remains a hard failure.

Stage 2 must be submitted on a CUDA node. It prefers the explicit Mellow
`mellow.model.htsat.HTSATWrapper` path, strictly loads the HTSAT backbone
after removing `sed_model.`, and records the intentionally uninitialized
Mellow `c2l` module separately. `--construction-only` never claims a forward
pass.

```bash
python code/RSmol/scripts/audit_audio_stage2_htsat_5_10_5_mellow.py \
  --mellow-root /hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main \
  --htsat-root /hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main \
  --htsat-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt \
  --audio-path /path/to/audio.wav --device cuda \
  --report-path /tmp/audio-stage2.json
```
