"""CPU-only: retain historical strongAC direct/CoT rows, omit weak/ReasonAQA."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from contract import (DEFAULT_DATA_DIR, DEFAULT_SOURCE, DEFAULT_RESOLUTIONS,
    PREVIOUS_SOURCE_COUNTS, STRONG_VIEW_LIMITS, SOURCE_COUNTS, digest, raw_references, sha256_file,
    validate_data_counts, write_json)
from manifest_io import iter_rows
from publication import atomic_publish

FORMAT = "mizar_l10_14x3_strong_a9gate_bias3_v1"
FILES = ("train.jsonl","selection.jsonl","resolved_paths.tsv","dataset_report.json")


def file_records(root, names):
    return {name:{"bytes":(root/name).stat().st_size,"sha256":sha256_file(root/name)} for name in names}


def verify_dataset(root):
    root=Path(root)
    report=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    body={k:v for k,v in report.items() if k!="seal_sha256"}
    if report.get("format")!=FORMAT or report.get("status")!="PASS" or report.get("seal_sha256")!=digest(body):
        raise ValueError("wrong strongAC dataset or invalid seal; regenerate with this recipe")
    validate_data_counts(report["rows"],report["source_counts"],report["selected_counts"])
    if report.get("audio_contribution_counts")!={"strong":report["rows"]}:
        raise ValueError("dataset must contain strong audio_contribution only")
    if set(report["files"])!=set(FILES):raise ValueError("incomplete dataset inventory")
    for name,expected in report["files"].items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=expected["bytes"] or sha256_file(p)!=expected["sha256"]:
            raise ValueError(f"dataset file changed: {p}")
    return report


def build(source, audit_tsv, output, *, expected_previous=None, expected_selected=None):
    """Validate full hybrid history, then filter exactly its strongAC views."""
    expected_previous=PREVIOUS_SOURCE_COUNTS if expected_previous is None else expected_previous
    # Retained strong counts are measured remotely, never assumed equal to the
    # original unfiltered S1 upper bound. Tests may require an exact tiny count.
    resolutions={}
    with Path(audit_tsv).open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle,delimiter="\t")
        if not {"field","raw_path","status","resolved_path"}<=set(reader.fieldnames or []):
            raise ValueError("invalid historical audit TSV")
        for record in reader:
            if record["status"]!="exists":continue
            key=(record["field"],record["raw_path"]);path=record["resolved_path"]
            if not path or (key in resolutions and resolutions[key]!=path):raise ValueError(f"conflicting historical path: {key}")
            resolutions[key]=path
    previous,selected,views,contributions=Counter(),Counter(),Counter(),Counter()
    omitted_weak=0
    used,stats,errors={}, {}, {}
    selected_hash=hashlib.sha256();kept=0
    with (output/"train.jsonl").open("w",encoding="utf-8",newline="\n") as train, \
         (output/"selection.jsonl").open("w",encoding="utf-8",newline="\n") as provenance:
        for index,row in enumerate(iter_rows(Path(source))):
            source_name=row.get("source_dataset") or row.get("source")
            previous[source_name]+=1
            if source_name=="reasonaqa_oneaudio":
                if row.get("source")!="reasonaqa_oneaudio":raise ValueError("contradictory ReasonAQA labels")
            else:
                if row.get("source")!="audiomcq" or source_name not in SOURCE_COUNTS:
                    raise ValueError(f"unexpected historical source at row {index}: {source_name}")
                strength=row.get("audio_contribution")
                if strength not in {"strong","weak"}:raise ValueError(f"unregistered audio_contribution at row {index}: {strength!r}")
                if strength!="strong":
                    omitted_weak+=1
                    if (index+1)%100000==0:
                        print(f"[manifest] scanned {index+1:,}; kept strongAC={kept:,}; omitted weak={omitted_weak:,}; omitted ReasonAQA={previous['reasonaqa_oneaudio']:,}",flush=True)
                    continue
                if row.get("stageE_variant") not in STRONG_VIEW_LIMITS:
                    raise ValueError(f"unexpected strongAC view at row {index}")
                if len(row.get("choices",[]))!=4 or row.get("gold_letter") not in {"a","b","c","d"}:
                    raise ValueError(f"invalid strongAC choices/gold at row {index}")
                if not all(isinstance(row.get(k),str) and row[k].strip() for k in ("input","answer")):
                    raise ValueError(f"empty input/answer at row {index}")
                _,refs=raw_references(row)
                if len(refs)!=1:raise ValueError("historical AudioMCQ requires one original recording")
                key=refs[0]
                if key not in used and key not in errors:
                    try:
                        if key not in resolutions:raise ValueError("previously used audio lacks a successful historical audit path")
                        p=Path(resolutions[key]).resolve(strict=True);stat=p.stat()
                        if not p.is_file() or stat.st_size<=0:raise ValueError("historical audio missing/empty")
                        used[key]=str(p);stats[str(p)]={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
                    except (OSError,ValueError) as exc:
                        errors[key]={"field":key[0],"raw_path":key[1],"previous_path":resolutions.get(key),"error":str(exc)}
                line=json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"
                train.write(line);selected_hash.update(line.encode("utf-8"))
                provenance.write(json.dumps({"selected_index":kept,"source_row_index":index,
                    "source_record_sha256":digest(row),"raw_field":key[0],"raw_path":key[1],
                    "resolved_path":used.get(key),"view":row.get("stageE_variant"),
                    "audio_contribution":row.get("audio_contribution")},ensure_ascii=False)+"\n")
                kept+=1;selected[source_name]+=1
                views[row.get("stageE_variant","UNCLASSIFIED")]+=1
                contributions[row.get("audio_contribution","UNCLASSIFIED")]+=1
            if (index+1)%100000==0:
                print(f"[manifest] scanned {index+1:,}; kept strongAC={kept:,}; omitted weak={omitted_weak:,}; omitted ReasonAQA={previous['reasonaqa_oneaudio']:,}; resolved files={len(stats):,}",flush=True)
        train.flush();os.fsync(train.fileno());provenance.flush();os.fsync(provenance.fileno())
    if previous!=expected_previous:raise ValueError(f"wrong historical hybrid inventory: {dict(previous)}; expected={expected_previous}")
    if expected_selected is not None and selected!=expected_selected:raise ValueError(f"unexpected retained strongAC inventory: {dict(selected)}")
    if not kept:raise ValueError("historical manifest contains no strongAC views")
    if errors:
        write_json(output/"unresolved_audio.json",list(errors.values()))
        raise ValueError(f"{len(errors)} previously trained strongAC references unavailable; restore original paths; see {output/'unresolved_audio.json'}")
    print(f"[manifest] rows written: {kept:,}; verifying {len(used):,} original references",flush=True)
    with (output/"resolved_paths.tsv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["field","raw_path","status","resolved_path"],delimiter="\t");writer.writeheader()
        for n,((field,raw),path) in enumerate(sorted(used.items()),1):
            stat=Path(path).stat()
            if {"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}!=stats[path]:raise ValueError(f"audio changed during prepare: {path}")
            writer.writerow({"field":field,"raw_path":raw,"status":"exists","resolved_path":path})
            if n%20000==0:print(f"[manifest] verified references {n:,}/{len(used):,}",flush=True)
    details={"rows":kept,"source_counts":dict(selected),"selected_counts":dict(views),
        "audio_contribution_counts":dict(contributions),"previous_source_counts":dict(previous),
        "omitted_reasonaqa":previous["reasonaqa_oneaudio"],"omitted_weak":omitted_weak,"unique_audio_files":len(stats),
        "unique_audio_bytes":sum(v["bytes"] for v in stats.values()),"rows_canonical_sha256":selected_hash.hexdigest(),
        "policy":"exact strongAC direct/CoT rows from historical hybrid in original order; no weak/ReasonAQA; exact historical audit paths",
        "audio_copied":False,"audio_decoded":False,"model_forward":False}
    write_json(output/"dataset_report.json",details)
    return details


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path(DEFAULT_SOURCE))
    p.add_argument("--resolved-paths",type=Path,default=Path(DEFAULT_RESOLUTIONS))
    p.add_argument("--output-dir",type=Path,default=Path(DEFAULT_DATA_DIR))
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args(argv);output=a.output_dir.resolve()
    if a.verify_only:print(json.dumps(verify_dataset(output),ensure_ascii=False));return
    inputs={"historical_hybrid_sha256":sha256_file(a.source),"historical_audit_sha256":sha256_file(a.resolved_paths),
            "preparer_sha256":sha256_file(Path(__file__)),"original_strong_source_bounds":SOURCE_COUNTS,
            "historical_source_counts":PREVIOUS_SOURCE_COUNTS}
    if output.exists():
        report=verify_dataset(output)
        if report["inputs"]!=inputs:raise ValueError("existing dataset has different inputs; use a new output directory")
        print(f"[manifest] reusing verified {output}",flush=True);return
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.with_name(output.name+".prepare.lock")
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);stage=None
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);fd=None
        stage=Path(tempfile.mkdtemp(prefix=output.name+".building-",dir=output.parent))
        details=build(a.source,a.resolved_paths,stage)
        if sha256_file(a.source)!=inputs["historical_hybrid_sha256"] or sha256_file(a.resolved_paths)!=inputs["historical_audit_sha256"]:
            raise ValueError("historical training inputs changed during preparation")
        validate_data_counts(details["rows"],details["source_counts"],details["selected_counts"])
        report={"format":FORMAT,"status":"PASS","inputs":inputs,"source_path":str(a.source.resolve()),
            "rows":details["rows"],"source_counts":details["source_counts"],"selected_counts":details["selected_counts"],
            "audio_contribution_counts":details["audio_contribution_counts"],
            "files":file_records(stage,FILES)}
        report["seal_sha256"]=digest(report);write_json(stage/"manifest.json",report);verify_dataset(stage)
        atomic_publish(stage,output);stage=None
        print(json.dumps({"event":"MANIFEST_PASS","rows":details["rows"],"source_counts":details["source_counts"],"views":details["selected_counts"],"dataset":str(output)},ensure_ascii=False),flush=True)
    except BaseException:
        if stage is not None:
            write_json(stage/"failure.json",{"status":"FAILED","traceback":traceback.format_exc()})
            print(f"[manifest] failed; diagnostics preserved at {stage}",flush=True)
        raise
    finally:
        if fd is not None:os.close(fd)
        lock.unlink()


if __name__=="__main__":main()
