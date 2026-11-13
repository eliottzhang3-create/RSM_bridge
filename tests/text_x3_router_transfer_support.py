"""Synthetic x2 donor metadata and explicit NumPy tensor fixtures; no production assets."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

from text_x3_test_support import BASE
sys.path.insert(0, str(BASE))
from text_5_10x3_5_mesh_6slot import router_transfer as transfer


def donor_metadata(path, width=16):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
    config.update(model_type='llama', hidden_size=width, num_hidden_layers=30,
                  recursive_layer_count=20, recursive_loops=2,
                  mesh_architecture_contract=transfer.SOURCE_ARCHITECTURE,
                  mesh_memory_slots=6, mesh_router_count=6, logical_to_physical=transfer.SOURCE_SCHEDULE)
    (path/'config.json').write_text(json.dumps(config))
    if not list(path.glob('model*.safetensors')):
        (path/'model.safetensors').write_bytes(b'header-reader fixture; not real tensors')
    (path/'training_state.pt').write_bytes(b'POISON: must never be deserialized')
    (path/'data_manifest.json').write_text('{}')
    (path/'mesh_checkpoint_metadata.json').write_text(json.dumps(dict(
        architecture_contract=transfer.SOURCE_ARCHITECTURE,
        training_contract=transfer.SOURCE_TRAINING_CONTRACT, precision='fp32', optimizer_step=9244,
        memory_slots=6, router_groups=3, router_module_count=6, router_parameters_in_optimizer=True,
        logical_to_physical=transfer.SOURCE_SCHEDULE)))
    files=sorted({p.name for p in path.iterdir() if p.is_file()}|{'checkpoint_manifest.json','checkpoint_complete.json'})
    record=dict(status='complete',architecture_contract=transfer.SOURCE_ARCHITECTURE,optimizer_step=9244,files=files)
    (path/'checkpoint_manifest.json').write_text(json.dumps(record))
    (path/'checkpoint_complete.json').write_text(json.dumps(dict(record,complete_marker=True)))
    return path


class Tensor:
    """Explicit fixture only; supports the real transfer/hash validation functions."""
    dtype='torch.float32'
    device='cpu'
    requires_grad=True
    def __init__(self, array): self.array=np.asarray(array,dtype=np.float32).copy()
    @property
    def shape(self): return self.array.shape
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array
    def isfinite(self): return np.isfinite(self.array)
    def untyped_storage(self): return NS(data_ptr=lambda:self.array.__array_interface__['data'][0])
    def copy_(self, value): self.array[:]=value.array


def source_tensors(width=16):
    return {key:Tensor(np.full((6,width) if key.endswith('.weight') else (6,), (i+1)/1000))
            for i,key in enumerate(sorted(set(transfer.router_tensor_mapping().values())))}


def provenance(identity=None):
    source_hashes=transfer.validate_router_tensors(source_tensors(),16)
    identity=identity or dict(checkpoint='/synthetic/checkpoint-009244',architecture_contract=transfer.SOURCE_ARCHITECTURE,
                             training_contract=transfer.SOURCE_TRAINING_CONTRACT,optimizer_step=9244,
                             hidden_size=16,memory_slots=6,router_groups=3,router_module_count=6)
    return dict(contract=transfer.ROUTER_INIT_CONTRACT,source=identity,tensor_mapping=transfer.router_tensor_mapping(),
                dtype='float32',source_tensor_sha256=source_hashes,
                target_tensor_sha256={k:source_hashes[v] for k,v in transfer.router_tensor_mapping().items()},
                routers_exact=True,independent_target_storage=True,load_source_lm=False,
                load_source_training_state=False,random_router_fallback=False)
