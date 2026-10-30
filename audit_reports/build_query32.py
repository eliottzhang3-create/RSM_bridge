from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine'
OLD = 'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn'
NEW = OLD.replace('layer36912_crossattn', 'layer36912_query32_crossattn')
SRC, DST = BASE / OLD, BASE / NEW
if DST.exists():
    raise SystemExit('new route already exists')
inventory = {str(p.relative_to(SRC)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(ROOT / 'audit_reports/query32_parent_before.json').write_text(json.dumps(inventory, indent=2))
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for path in DST.rglob('*'):
    if path.suffix in {'.py', '.sh', '.md', '.json'}:
        value = path.read_text(encoding='utf-8')
        value = value.replace('layer36912_crossattn', 'layer36912_query32_crossattn')
        value = value.replace('multilayer_crossattn', 'query32_crossattn')
        if path.name == 'contract.py':
            value = value.replace('query32_crossattn_v2', 'query32_crossattn_v1')
        path.write_text(value, encoding='utf-8', newline='\n')
launcher = ROOT / 'code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_crossattn_5090.sh'
target = launcher.with_name(launcher.name.replace('layer36912_crossattn', 'layer36912_query32_crossattn'))
target.write_text(launcher.read_text(encoding='utf-8').replace('layer36912_crossattn', 'layer36912_query32_crossattn')
                  .replace('mizar-ced-xattn-', 'mizar-ced-q32-')
                  .replace('layers3/6/9/12 cross attention', 'layers3/6/9/12 shared query32 cross attention'),
                  encoding='utf-8', newline='\n')
for suffix in ('route', 'behavior', 'tensor', 'shared_start'):
    src = ROOT / f'tests/test_mizar_s1_ced_multilayer_crossattn_{suffix}.py'
    target = ROOT / f'tests/test_mizar_s1_ced_query32_crossattn_{suffix}.py'
    value = src.read_text(encoding='utf-8').replace('layer36912_crossattn', 'layer36912_query32_crossattn')
    value = value.replace('multilayer_crossattn', 'query32_crossattn').replace('4892741', '6241733')
    if suffix == 'route':
        value = value.replace('contract.ROUTE.endswith("_v2")', 'contract.ROUTE.endswith("_query32_crossattn_v1")')
        value = value.replace('contract.ROUTE[:-1]+"1"', 'contract.ROUTE[:-1]+"2"')
    target.write_text(value, encoding='utf-8', newline='\n')
src = ROOT / 'tests/validate_mizar_s1_ced_multilayer_crossattn.py'
value = src.read_text(encoding='utf-8').replace('layer36912_crossattn', 'layer36912_query32_crossattn')
value = value.replace('multilayer_crossattn', 'query32_crossattn')
# Run the direct v2 parent as a separate process too, keeping module namespaces isolated.
value = value.replace('FILES = [', 'FILES = ["test_mizar_s1_ced_multilayer_crossattn_behavior.py",\n'
                      '         "test_mizar_s1_ced_multilayer_crossattn_route.py",\n'
                      '         "test_mizar_s1_ced_multilayer_crossattn_tensor.py",\n')
(ROOT / 'tests/validate_mizar_s1_ced_query32_crossattn.py').write_text(value, encoding='utf-8', newline='\n')
print(DST)
