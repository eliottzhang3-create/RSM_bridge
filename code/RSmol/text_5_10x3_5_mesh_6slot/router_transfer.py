"""Route-specific router transfer identity. JSON checks do not load training state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_ROUTER_CHECKPOINT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_6slot/formal_1epoch_fp32_20261031_124955519529617/checkpoint-009244')
ROUTER_INIT_CONTRACT = 'trained_x2_6slot_routers_to_x3_pre_0_0_1_pre_pre_0_1_v1'
SOURCE_ARCHITECTURE = 'logical_30_physical_20_5_10x2_5_mesh_6slot_3router'
SOURCE_TRAINING_CONTRACT = 'text_mesh_x2_6slot_fp32_full_epoch_v1'
SOURCE_SCHEDULE = [*range(5), *list(range(5, 15)) * 2, *range(15, 20)]
ROUTER_GROUP_MAP = {'write': (0, 1, 1, 2), 'read': (0, 0, 1, 2)}


def router_tensor_mapping() -> dict[str, str]:
    return {f'model.{kind}_routers.{target}.{part}': f'model.{kind}_routers.{source}.{part}'
            for kind, groups in ROUTER_GROUP_MAP.items() for target, source in enumerate(groups)
            for part in ('weight', 'bias')}


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
    result.update(value.numpy().tobytes(order='C'))
    return result.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'router source JSON must contain an object: {path}')
    return value


def validate_router_checkpoint(path: Path, hidden_size: int) -> dict[str, Any]:
    """Validate the donor through JSON and file inventory, never training_state.pt."""
    path = path.expanduser().resolve(strict=True)
    if not path.is_dir() or path.name != 'checkpoint-009244':
        raise ValueError('router donor must be the complete x2 six-slot checkpoint-009244')
    config = _json_object(path / 'config.json')
    expected_config = dict(model_type='llama', hidden_size=hidden_size, num_hidden_layers=30,
                           recursive_layer_count=20, recursive_loops=2,
                           mesh_architecture_contract=SOURCE_ARCHITECTURE,
                           mesh_memory_slots=6, mesh_router_count=6,
                           logical_to_physical=SOURCE_SCHEDULE)
    if any(config.get(k) != v for k, v in expected_config.items()):
        raise ValueError('router donor must have the matching x2 six-slot architecture and hidden size')
    metadata = _json_object(path / 'mesh_checkpoint_metadata.json')
    expected_metadata = dict(architecture_contract=SOURCE_ARCHITECTURE,
                             training_contract=SOURCE_TRAINING_CONTRACT, precision='fp32',
                             optimizer_step=9244, memory_slots=6, router_groups=3,
                             router_module_count=6, router_parameters_in_optimizer=True,
                             logical_to_physical=SOURCE_SCHEDULE)
    if any(metadata.get(k) != v for k, v in expected_metadata.items()):
        raise ValueError('router donor checkpoint metadata is incompatible')
    manifest = _json_object(path / 'checkpoint_manifest.json')
    complete = _json_object(path / 'checkpoint_complete.json')
    for record in (manifest, complete):
        if any(record.get(k) != v for k, v in dict(status='complete', architecture_contract=SOURCE_ARCHITECTURE, optimizer_step=9244).items()):
            raise ValueError('router donor checkpoint completion identity mismatch')
    files = manifest.get('files')
    if (complete.get('complete_marker') is not True or not isinstance(files, list) or not files
            or complete.get('files') != files
            or any(not isinstance(name, str) or not name or '/' in name or '\\' in name
                   or name in {'.', '..'} or not (path / name).is_file() for name in files)
            or len(set(files)) != len(files)):
        raise ValueError('router donor checkpoint has an incomplete or unsafe manifest')
    required = {'config.json', 'mesh_checkpoint_metadata.json', 'checkpoint_manifest.json',
                'checkpoint_complete.json', 'training_state.pt', 'data_manifest.json'}
    if not required.issubset(files):
        raise ValueError('router donor checkpoint manifest lacks required files')
    weights = sorted(p.name for p in path.glob('model*.safetensors'))
    if not weights or not set(weights).issubset(files):
        raise ValueError('router donor requires manifested model safetensors files')
    return dict(checkpoint=str(path), architecture_contract=SOURCE_ARCHITECTURE,
                training_contract=SOURCE_TRAINING_CONTRACT, optimizer_step=9244,
                hidden_size=hidden_size, memory_slots=6, router_groups=3, router_module_count=6,
                files_sha256={name: file_sha256(path / name) for name in
                    ('config.json', 'mesh_checkpoint_metadata.json', 'checkpoint_manifest.json', 'checkpoint_complete.json')},
                weight_files=[dict(name=name, bytes=(path / name).stat().st_size,
                                   sha256=file_sha256(path / name)) for name in weights])


def validate_router_tensors(tensors: dict[str, Any], hidden_size: int) -> dict[str, str]:
    expected = set(router_tensor_mapping().values())
    if set(tensors) != expected:
        raise ValueError('router donor must provide exactly twelve x2 router weight/bias tensors')
    for name, value in tensors.items():
        shape = (6, hidden_size) if name.endswith('.weight') else (6,)
        if tuple(value.shape) != shape or str(value.dtype) != 'torch.float32':
            raise ValueError(f'router donor shape/dtype mismatch: {name}; expected FP32 {shape}')
        if not bool(value.isfinite().all().item()):
            raise ValueError(f'nonfinite router donor tensor: {name}')
    return {name: tensor_sha256(value) for name, value in sorted(tensors.items())}


def validate_transfer_provenance(record: Any) -> dict[str, Any]:
    """Check saved conversion provenance without requiring donor assets at training time."""
    if not isinstance(record, dict):
        raise ValueError('trained router transfer provenance is missing')
    expected = dict(contract=ROUTER_INIT_CONTRACT, tensor_mapping=router_tensor_mapping(),
                    dtype='float32', routers_exact=True, independent_target_storage=True,
                    load_source_lm=False, load_source_training_state=False, random_router_fallback=False)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('trained router transfer contract or mapping mismatch')
    source = record.get('source', {})
    if not isinstance(source, dict) or any(source.get(k) != v for k, v in dict(
            architecture_contract=SOURCE_ARCHITECTURE, training_contract=SOURCE_TRAINING_CONTRACT,
            optimizer_step=9244, memory_slots=6, router_groups=3, router_module_count=6).items()):
        raise ValueError('trained router transfer donor identity mismatch')
    if not isinstance(source.get('checkpoint'), str) or Path(source['checkpoint']).name != 'checkpoint-009244':
        raise ValueError('trained router transfer donor path mismatch')
    source_hashes = record.get('source_tensor_sha256', {})
    target_hashes = record.get('target_tensor_sha256', {})
    mapping = router_tensor_mapping()
    def valid_hashes(values, keys):
        return isinstance(values, dict) and set(values) == set(keys) and all(
            isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in values.values())
    if not valid_hashes(source_hashes, mapping.values()) or not valid_hashes(target_hashes, mapping):
        raise ValueError('trained router transfer tensor hashes are incomplete')
    if any(target_hashes[target] != source_hashes[original] for target, original in mapping.items()):
        raise ValueError('trained router transfer hashes do not match the declared mapping')
    return record


def validate_materialized_routers(model: Any, provenance: dict[str, Any]) -> None:
    """Actual target values/storage must agree, including after a saved model reload."""
    validate_transfer_provenance(provenance)
    mapping = router_tensor_mapping()
    parameters = dict(model.named_parameters())
    names = {name for name in parameters if '.write_routers.' in name or '.read_routers.' in name}
    if names != set(mapping):
        raise ValueError('target model must contain exactly sixteen transferred router tensors')
    seen = set()
    for name in mapping:
        parameter = parameters[name]
        storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
        if (storage in seen or not parameter.requires_grad or str(parameter.dtype) != 'torch.float32'
                or tensor_sha256(parameter) != provenance['target_tensor_sha256'][name]):
            raise ValueError(f'target router values, dtype, storage or trainability mismatch: {name}')
        seen.add(storage)
