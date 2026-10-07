# ReasonAQA x4 two-loop inference ablation

This isolated generation route loads the x4/7-slot audio checkpoint
`formal_3epochs_20261002_configfix_v3/checkpoint-011343`. It selects the first
five rows (zero-based indices 0–4) from `/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json`,
matching the sample selection in `run_reasonaqa_mellow_official_training_generation_3090.sh`.
The official Mellow model and its runtime YAML are not used.

Inference executes the five prefix layers and two ten-layer middle loops.
The second loop writes with `write_routers[2]`, then `read_routers[4]` reads
the current memory using the second loop's input as its query. The final five
layers execute next. The third and fourth middle loops are skipped. The
checkpoint weights and saved configuration remain unchanged.

After syncing code to the server, submit from `code/RSmol`:

```bash
bash run_reasonaqa_x4_two_loop_ablation_generation_3090.sh
```

Pass `--sample-indices 3 8 12` to choose other zero-based test rows. Pass
`--compare-full` to also generate full x4 outputs for the same audio and
prompt. Outputs are `samples.jsonl` and `generation_report.json` in a unique
timestamped directory under `outputs/RSmol/reasonaqa_x4_two_loop_ablation`.
