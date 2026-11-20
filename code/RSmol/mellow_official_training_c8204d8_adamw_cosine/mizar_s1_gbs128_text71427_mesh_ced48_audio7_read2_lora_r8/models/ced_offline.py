"""Load the original CED config without its hidden Hub label-table dependency.

Upstream CedConfig unconditionally reads an external AudioSet CSV, even when
the local JSON already includes the complete labels. Supply that same metadata
through a temporary CSV while loading the original config and model classes.
Only the CED configuration module's imported cached_file alias is scoped here;
the global Transformers/Hub functions and pretrained files are never changed.
"""
from contextlib import contextmanager
import csv
import importlib
import json
from pathlib import Path
import tempfile

from contract import ced_label_metadata, validate_ced_config


def _load_config_class(model_dir):
    # Keep this module importable by stdlib regression tests without torch.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    return get_class_from_dynamic_module(
        "configuration_ced.CedConfig", str(model_dir), local_files_only=True)


@contextmanager
def local_ced_config(model_dir):
    model_dir = Path(model_dir).resolve()
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    validate_ced_config(raw)
    labels, reverse = ced_label_metadata(raw)
    config_class = _load_config_class(model_dir)
    module = importlib.import_module(config_class.__module__)
    missing = object()
    original = getattr(module, "cached_file", missing)
    with tempfile.TemporaryDirectory(prefix="mizar-ced-labels-") as directory:
        table = Path(directory) / "class_labels_indices.csv"
        with table.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("index", "mid", "display_name"))
            # The constructor only reads columns 0 and 2. No ontology IDs are
            # invented: the unused mid column stays empty.
            writer.writerows((index, "", label) for index, label in labels.items())

        def local_label_file(repository, filename, *args, **kwargs):
            if (repository, filename) != ("topel/ConvNeXt-Tiny-AT", "class_labels_indices.csv"):
                raise FileNotFoundError(f"unsupported external CED config dependency: {repository}/{filename}")
            return str(table)

        module.cached_file = local_label_file
        try:
            config = config_class.from_pretrained(str(model_dir), local_files_only=True)
            # Upstream's CSV loop drops duplicate names. Preserve every index
            # from JSON, including the genuine repeated AudioSet display names.
            config.id2label = dict(labels)
            config.label2id = dict(reverse)
            validate_ced_config(config.to_dict())
            yield config
        finally:
            if original is missing:
                del module.cached_file
            else:
                module.cached_file = original
