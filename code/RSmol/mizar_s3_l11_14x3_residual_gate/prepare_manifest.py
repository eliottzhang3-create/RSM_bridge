"""Read-only verification of the already prepared S3 dataset; never rebuild."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mizar_s3_data_prepare.prepare import verify_dataset as verify_s3
from contract import validate_data_counts

def verify_dataset(root):
    manifest=verify_s3(root)
    report=json.loads((Path(root)/"dataset_report.json").read_text(encoding="utf-8"))
    validate_data_counts(manifest["rows"],report["kept_by_source"])
    return {**manifest,"source_counts":report["kept_by_source"],"inputs":manifest["binding"]["inputs"],
            "selected_counts":report["kept_by_source"]}
