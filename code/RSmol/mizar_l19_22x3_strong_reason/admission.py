"""CPU admission before vc submit: dataset/index/release/full-resume headers."""
import argparse
import json
from pathlib import Path
from contract import ROUTE, digest, write_json
from data.prepared_index import load_index
from prepare_manifest import verify_dataset
from write_config import build_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True,type=Path)
    p.add_argument("--index-dir",required=True,type=Path)
    p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--output-root",required=True,type=Path)
    p.add_argument("--resume-checkpoint",type=Path)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args()
    data=verify_dataset(a.data_dir)
    prepared=load_index(a.index_dir,a.data_dir/"train.jsonl",a.data_dir/"resolved_paths.tsv")
    contract=build_contract(prepared,a.model_dir)
    step=0
    if a.resume_checkpoint:
        if a.resume_checkpoint.resolve().parent.parent.parent != a.output_root.resolve():
            raise ValueError("full resume must use the original outer run root")
        import torch
        saved=torch.load(a.resume_checkpoint,map_location="cpu",weights_only=False,mmap=True)
        step=saved.get("total_step")
        if (saved.get("schema_version")!=1 or saved.get("route_variant")!=ROUTE
                or saved.get("mode")!="formal" or saved.get("contract")!=contract
                or saved.get("contract_sha256")!=digest(contract)
                or type(step) is not int or not 1<=step<contract["geometry"]["total_steps"]
                or (saved.get("epoch"),saved.get("next_optimizer_step"))!=(0,step)
                or len(saved.get("random_state_by_rank",[]))!=8):
            raise ValueError("resume header/route/source/data/environment/recipe/cursor mismatch")
        later=[p for p in a.resume_checkpoint.parent.glob("model--step-*.ckpt") if int(p.stem.rsplit("-",1)[1])>step]
        if later:
            raise ValueError(f"newer complete checkpoints already exist; choose the newest valid checkpoint: {later}")
        del saved
    elif a.output_root.exists() and any(a.output_root.iterdir()):
        raise ValueError("fresh training requires a new empty output root")
    report={"status":"READY_FOR_GPU","route":ROUTE,"dataset_seal":data["seal_sha256"],
            "contract_sha256":digest(contract),"geometry":contract["geometry"],"resume_step":step,
            "output_root":str(a.output_root),"model_dir":str(a.model_dir),"gpu_forward":False}
    write_json(a.report,report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
