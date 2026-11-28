"""Collect per-rank diagnostics when torchrun exits nonzero."""
import json
from pathlib import Path
import sys
from contract import write_json
run=Path(sys.argv[1]);status=int(sys.argv[2])
phase=sys.argv[3] if len(sys.argv)>3 else "torchrun"
config=json.loads((run/"runtime.json").read_text(encoding="utf-8")) if (run/"runtime.json").exists() else {}
failures=[]
for path in sorted(run.glob("failure_rank*.json")):
    record=json.loads(path.read_text(encoding="utf-8"))
    if phase=="torchrun" and record.get("attempt_id")==config.get("attempt_id"):
        failures.append(record)
staging=json.loads((run/"staging_progress.json").read_text(encoding="utf-8")) if (run/"staging_progress.json").exists() else None
write_json(run/"training_report.json",{"status":"FAILED","exit_code":status,"shell_phase":phase,"staging":staging,
           "attempt_id":config.get("attempt_id"),"rank_failures":failures,
           "note":"Inspect the earliest rank traceback; torchrun/NCCL summaries may be secondary."})
