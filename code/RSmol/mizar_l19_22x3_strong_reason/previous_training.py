"""Prove prior training membership before excluding any strongAC view."""
from collections import Counter
from pathlib import Path
from contract import digest, raw_references, sha256_file
from manifest_io import iter_rows

PREVIOUS_SOURCE_COUNTS = {
    "reasonaqa_oneaudio":756968, "AudioCaps":113844, "Clotho":4798,
    "MusicCaps":3221, "CompA-R":249161, "LP-MusicCaps-MTT":19632,
    "SpeechCraft":287727, "TACOS":42141,
}
DEFAULT_PREVIOUS_MANIFEST = "/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json"


class PreviousTraining:
    def __init__(self, path, resolver, *, expected_source_counts=None):
        from prepare_manifest import row_kind
        self.path = Path(path).resolve(strict=True)
        self.sha256 = sha256_file(self.path)
        self.references, self.strong_records = set(), Counter()
        self.strong_counts, self.source_counts = Counter(), Counter()
        self.rows = 0
        for row in iter_rows(self.path):
            self.rows += 1
            source, refs = raw_references(row)
            self.source_counts[source] += 1
            self.references.update(refs)
            kind = row_kind(row)
            if kind in ("audiomcq_mcq", "audiomcq_cot"):
                self.strong_records[digest(row)] += 1
                self.strong_counts[kind] += 1
            if self.rows % 100000 == 0:
                print(f"[previous-training] scanned {self.rows:,}; strongAC views={sum(self.strong_counts.values()):,}", flush=True)
        expected = PREVIOUS_SOURCE_COUNTS if expected_source_counts is None else expected_source_counts
        if self.source_counts != expected or self.rows != sum(expected.values()):
            raise ValueError(f"previous training manifest counts differ: {dict(self.source_counts)}; expected={expected}")
        # A previous training row without its successful audit is inconsistent
        # evidence. Never convert this into permission to drop a used sample.
        absent = sorted(self.references - set(resolver.resolutions))
        if absent:
            raise ValueError(f"previous training used {len(absent)} references without an exists audit record; first={absent[:5]}")
        if sha256_file(self.path) != self.sha256:
            raise ValueError("previous training manifest changed while reading")
        self.strong_total = sum(self.strong_counts.values())
        if self.strong_total <= 0:
            raise ValueError("previous training contains no strongAC views")
        resolver.required_previous_references = self.references

    def describe(self):
        return {"path":str(self.path), "sha256":self.sha256, "rows":self.rows,
                "source_counts":dict(self.source_counts), "strong_counts":dict(self.strong_counts),
                "strong_views":self.strong_total,
                "policy":"retain exact original strongAC rows used by previous hybrid training; reuse its successful audited paths; never filter newly unavailable used audio"}


def load_previous_training(path, resolver, **kwargs):
    return PreviousTraining(path, resolver, **kwargs)
