"""Select original shared audio until the trainer switches all ranks to local."""
from pathlib import Path


class AudioPaths:
    def __init__(self, stage_root, read_policy="staged_only"):
        if read_policy not in {"staged_only", "shared_until_complete"}:
            raise ValueError("unknown audio read policy")
        self.root = Path(stage_root).resolve()
        self.local = read_policy == "staged_only"
        self.read_files = {"shared": 0, "local": 0}

    def use_local(self):
        self.local = True

    def resolve(self, row):
        local_names = row["s1_raw_paths"]
        if len(local_names) not in {1, 2}:
            raise ValueError("expected one or two original recordings")
        if self.local:
            paths = [(self.root / name).resolve() for name in local_names]
            if any(not path.is_relative_to(self.root) or path.suffix == ".partial" for path in paths):
                raise ValueError("audio path escaped the job's staging directory or names a partial file")
            origin = "local"
        else:
            paths = [Path(name) for name in row["s1_original_paths"]]
            if len(paths) != len(local_names) or any(not path.is_absolute() for path in paths):
                raise ValueError("invalid CPU-indexed original audio paths")
            origin = "shared"
        self.read_files[origin] += len(paths)
        return paths
