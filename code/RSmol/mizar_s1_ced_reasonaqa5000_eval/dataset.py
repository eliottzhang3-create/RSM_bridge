"""Filter original JSON order and resolve only selected ReasonAQA audio paths."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath

from . import storage

CATEGORIES = ('audiocaps_mcq', 'clotho_mcq', 'clotho_aqa_yesno')
SUBTYPES = {'AudioCaps-MCQ.json': CATEGORIES[0], 'Clotho-MCQ.json': CATEGORIES[1]}
SELECTION_POLICY = {'mcq_subtypes': list(SUBTYPES), 'yesno_subtype': 'ClothoAQA.json',
                    'yesno_answers_after_strip_casefold': ['yes', 'no'],
                    'order': 'original_test_json_row_order_no_deduplication'}


def select_records(path):
    raw = path.read_bytes()
    rows = json.loads(raw.decode('utf-8'))
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError('ReasonAQA test.json must be a nonempty list of objects')
    selected, excluded, source_subtypes = [], Counter(), Counter()
    for index, row in enumerate(rows):
        subtype = row.get('subtype')
        source_subtypes[str(subtype)] += 1
        category = SUBTYPES.get(subtype) if isinstance(subtype, str) else None
        if subtype == 'ClothoAQA.json':
            answer = row.get('answer')
            if isinstance(answer, str) and answer.strip().casefold() in ('yes', 'no'):
                category = CATEGORIES[2]
            else:
                excluded['ClothoAQA_non_yesno'] += 1
                continue
        if category is None:
            excluded[str(subtype)] += 1
            continue
        for field in ('input', 'answer', 'filepath1', 'filepath2'):
            if not isinstance(row.get(field), str) or (field != 'filepath2' and not row[field].strip()):
                raise ValueError(f'invalid selected row {index}: {field}')
        selected.append({'id': f'reasonaqa_test:{index}', 'source_row_index': index,
                         'category': category, 'original_record': row})
    if not selected:
        raise ValueError('no ReasonAQA close-ended rows selected')
    audit = {'source_rows': len(rows), 'selected_rows': len(selected),
             'source_subtype_counts': dict(sorted(source_subtypes.items())),
             'selected_category_counts': {key: sum(r['category'] == key for r in selected) for key in CATEGORIES},
             'excluded_counts': dict(sorted(excluded.items())), 'selection_policy': SELECTION_POLICY,
             'selection_sha256': storage.object_sha(selected)}
    return selected, audit


def normalized_stem(value):
    stem = PurePosixPath(value).stem
    stem = unicodedata.normalize('NFKD', stem).encode('ascii', 'ignore').decode('ascii')
    return re.sub('[^a-z0-9]+', '', stem.casefold())


class AudioResolver:
    def __init__(self, args):
        self.roots = {CATEGORIES[0]: args.audiocaps_root, CATEGORIES[1]: args.clotho_root,
                      CATEGORIES[2]: args.clotho_aqa_root}
        self.cache = {}
        self.aqa_index = None

    def resolve(self, raw, category):
        logical = raw.strip().replace('\\', '/')
        key = (logical, category)
        if key in self.cache:
            return self.cache[key]
        parts = PurePosixPath(logical).parts
        if not parts or '..' in parts or re.match(r'^[A-Za-z]:', logical):
            raise ValueError(f'unsafe logical audio path: {raw!r}')
        root = self.roots[category].resolve()
        if logical.startswith('/'):
            candidates = [Path(logical)]
        elif category == CATEGORIES[2]:
            candidates = [root / parts[-1], root.joinpath(*parts[1:])]
        else:
            markers = ('train', 'val', 'test') if category == CATEGORIES[0] else ('development', 'validation', 'evaluation')
            split = next((i for i, part in enumerate(parts) if part in markers), None)
            candidates = [root.joinpath(*parts[split:])] if split is not None else []
            candidates.append(root.joinpath(*(parts[1:] if len(parts) > 1 else parts)))
        matches = {p.resolve() for p in candidates if p.is_file()}
        if not matches and category == CATEGORIES[2] and root.is_dir():
            # Historical ClothoAQA filenames sometimes differ in punctuation/Unicode.
            # Build this filenames-only index lazily, and reject any ambiguity.
            if self.aqa_index is None:
                self.aqa_index = {}
                for path in root.rglob('*'):
                    if path.is_file() and path.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg'):
                        self.aqa_index.setdefault(normalized_stem(path.name), set()).add(path.resolve())
            matches = self.aqa_index.get(normalized_stem(logical), set())
        if len(matches) != 1:
            raise FileNotFoundError(f'audio missing or ambiguous: {raw!r}; matches={sorted(map(str, matches))}; candidates={list(map(str, candidates))}')
        path = next(iter(matches))
        if not path.is_relative_to(root):
            raise ValueError(f'audio path outside configured category root: {path}; root={root}')
        stat = path.stat()
        value = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        self.cache[key] = value
        return value


def prepare_records(records, args):
    resolver = AudioResolver(args)
    result = []
    for record in records:
        row, category = record['original_record'], record['category']
        one = resolver.resolve(row['filepath1'], category)
        two = resolver.resolve(row['filepath2'], category) if row['filepath2'].strip() else None
        result.append({**record, 'audio1': one, 'audio2': two})
    return result
