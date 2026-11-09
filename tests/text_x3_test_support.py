"""Load real route helpers without importing unavailable training dependencies."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'code/RSmol'
ROUTES = ('5_10x3_5', '5_10x3_5_mesh_6slot')
SCHEDULE = [*range(5), *list(range(5, 15)) * 3, *range(15, 20)]
SOURCE_MAPPING = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)


def model_constants(route):
    tree = ast.parse((BASE / f'recursive_model_{route}.py').read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<model constants>', 'exec'), namespace)
    return namespace


def helpers(route, converter=False):
    path = BASE / 'scripts' / (f'convert_stepwise_{route}.py' if converter else f'train_stage4_{route}_ddp.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name.startswith('torch') for alias in node.names)
        or isinstance(node, ast.ImportFrom) and (node.module or '').startswith('recursive_model_'))]
    tree.body = [node for node in tree.body if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')]
    module = types.ModuleType(f'_x3_helpers_{route}_{converter}')
    module.__file__ = str(path)
    module.__dict__.update(model_constants(route))
    module.torch = types.SimpleNamespace()
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
