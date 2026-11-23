"""Read-only offline regrouping: weighted counts, identity checks and actual PNG."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage
from mizar_ablation_regroup import regroup


COUNTS = [53,49,35,46,34,33,33,48, 44,22,21,33,46,34,20,10, 48,48,56,29,35,20,27,33,48,48,47]


def make_source(source):
    mapping = regroup.read_mapping()
    names = [name for group in regroup.GROUPS for name in mapping['groups'][group]['subcategories']]
    expected = storage.seal({'task':'ablation', 'protocol':{'rows':1000, 'conditions':list(regroup.CONDITIONS)},
                             'fixture':'EXPLICIT_SYNTHETIC_COUNTS_NOT_MODEL_RESULTS'}, 'identity_sha256')
    storage.write_json(source/'run_config.json', expected)
    raw_base = storage.object_sha({'fixture':'baseline raw rows not loaded by this CPU aggregator'})
    summaries = []
    baseline_correct = {name:(index*7+5) % (count+1) for index,(name,count) in enumerate(zip(names, COUNTS))}
    for layer in regroup.CONDITIONS:
        groups = {}
        for index, (name, total) in enumerate(zip(names, COUNTS)):
            correct = baseline_correct[name] if layer is None else (index*7+5+layer) % (total+1)
            groups[name] = {'correct':correct, 'total':total, 'accuracy':correct/total,
                            'delta_vs_baseline':(correct-baseline_correct[name])/total}
        correct = sum(row['correct'] for row in groups.values())
        baseline_total = sum(baseline_correct.values())
        summary = {'condition':regroup.condition_name(layer), 'drop_layer_1based':layer,
                   'correct':correct, 'total':1000, 'accuracy':correct/1000,
                   'baseline_accuracy':baseline_total/1000, 'delta_vs_baseline':(correct-baseline_total)/1000,
                   'groups':{'sub_category':groups}, 'identity_sha256':expected['identity_sha256'],
                   'raw_rows_sha256':raw_base if layer is None else storage.object_sha({'fixture_layer':layer}),
                   'baseline_rows_sha256':raw_base}
        storage.write_json(source/'conditions'/regroup.condition_name(layer)/'summary.json', storage.seal(summary))
        summaries.append(summary)
    storage.write_json(source/'condition_scores.json', summaries)
    storage.write_json(source/'baseline.json', summaries[0])
    storage.write_json(source/'analysis_report.json', {'status':'PASS', 'conditions_completed':29, 'decisions':29000,
        'rows_per_condition':1000, 'identity_sha256':expected['identity_sha256']})
    return mapping, summaries, expected


def rewrite_condition(source, index, summary):
    values = storage.read_json(source/'condition_scores.json')
    values[index] = summary
    storage.write_json(source/'condition_scores.json', values)
    storage.write_json(source/'conditions'/summary['condition']/'summary.json', storage.seal(summary))
    if index == 0:
        storage.write_json(source/'baseline.json', summary)


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)


class RegroupTests(TempCase):
    def test_exact_user_mapping_full27_and_real_metadata_counts(self):
        mapping, summaries, _ = make_source(self.path/'source')
        self.assertEqual([len(mapping['groups'][g]['subcategories']) for g in regroup.GROUPS], [8,8,11])
        rows = regroup.aggregate(summaries, mapping)
        self.assertEqual([rows[0]['groups'][g]['total'] for g in regroup.GROUPS], [331,230,439])
        self.assertEqual(len(rows), 29)
        for row in rows:
            self.assertEqual(sum(g['total'] for g in row['groups'].values()), 1000)
        local = Path(r'C:\Xlance\assets\Mizar\Mizar_159M-main\Mizar_159M-main\data\evaluation\mmau_mini_official.json')
        if local.exists():
            from collections import Counter
            counts = Counter(row['sub-category'] for row in storage.read_json(local))
            expected = {name:n for group in mapping['groups'].values() for name,n in
                        [(name,counts[name]) for name in group['subcategories']]}
            self.assertEqual(set(counts),set(expected))
            self.assertEqual([sum(counts[name] for name in mapping['groups'][g]['subcategories']) for g in regroup.GROUPS], [331,230,439])

    def test_count_weighted_accuracy_and_percentage_point_delta_not_macro_mean(self):
        mapping, summaries, _ = make_source(self.path/'source')
        rows = regroup.aggregate(summaries, mapping)
        for i, row in enumerate(rows):
            for group in regroup.GROUPS:
                names = mapping['groups'][group]['subcategories']
                correct = sum(summaries[i]['groups']['sub_category'][name]['correct'] for name in names)
                total = sum(summaries[i]['groups']['sub_category'][name]['total'] for name in names)
                saved = row['groups'][group]
                self.assertAlmostEqual(saved['accuracy'],correct/total)
                self.assertAlmostEqual(saved['delta_percentage_points'],100*(correct-saved['baseline_correct'])/total)
        names = mapping['groups']['perception']['subcategories']
        macro = sum(summaries[0]['groups']['sub_category'][name]['accuracy'] for name in names)/len(names)
        self.assertNotAlmostEqual(rows[0]['groups']['perception']['accuracy'],macro)

    def test_loader_requires_completed_source_and_exact29_conditions(self):
        source = self.path/'source'
        mapping, _, _ = make_source(source)
        regroup.load_completed(source,mapping)
        report = storage.read_json(source/'analysis_report.json')
        for edit in ({'status':'FAILED'},{'decisions':28000},{'conditions_completed':28},{'identity_sha256':'other'}):
            storage.write_json(source/'analysis_report.json',{**report,**edit})
            with self.assertRaises(ValueError): regroup.load_completed(source,mapping)
        storage.write_json(source/'analysis_report.json',report)
        values = storage.read_json(source/'condition_scores.json')
        storage.write_json(source/'condition_scores.json',values[:-1])
        with self.assertRaises(ValueError): regroup.load_completed(source,mapping)

    def test_hash_tampering_or_summary_disagreement_rejected(self):
        source = self.path/'source'; mapping, summaries, _ = make_source(source)
        path = source/'conditions/ablate_02/summary.json'
        tampered = storage.read_json(path); tampered['correct'] += 1
        storage.write_json(path,tampered)
        with self.assertRaisesRegex(ValueError,'content hash'): regroup.load_completed(source,mapping)
        storage.write_json(path,storage.seal(summaries[1]))
        different = {**summaries[1], 'elapsed_seconds':1}
        storage.write_json(path,storage.seal(different))
        with self.assertRaisesRegex(ValueError,'differs from combined'): regroup.load_completed(source,mapping)

    def test_unknown_missing_subcategory_and_changed_denominator_rejected(self):
        source = self.path/'source'; mapping, summaries, _ = make_source(source)
        original = summaries[1]
        for mode in ('unknown','missing','counts'):
            value = copy.deepcopy(original)
            subsets = value['groups']['sub_category']
            first = next(iter(subsets))
            if mode == 'unknown': subsets['unmapped_category'] = subsets.pop(first)
            if mode == 'missing': subsets.pop(first)
            if mode == 'counts':
                subsets[first]['total'] += 1
                subsets[first]['accuracy'] = subsets[first]['correct']/subsets[first]['total']
            rewrite_condition(source,1,value)
            with self.assertRaises(ValueError): regroup.load_completed(source,mapping)

    def test_bad_accuracy_and_other_baseline_reference_rejected(self):
        source = self.path/'source'; mapping, summaries, _ = make_source(source)
        for edit in ({'accuracy':.123456},{'baseline_rows_sha256':'other'}, {'delta_vs_baseline':.987}):
            rewrite_condition(source,1,{**summaries[1],**edit})
            with self.assertRaises(ValueError): regroup.load_completed(source,mapping)

    def test_duplicate_group_mapping_rejected(self):
        mapping = regroup.read_mapping()
        mapping['groups']['semantic']['subcategories'][0] = mapping['groups']['perception']['subcategories'][0]
        from unittest.mock import patch
        with patch.object(storage,'read_json',return_value=mapping), self.assertRaisesRegex(ValueError,'more than one group'):
            regroup.read_mapping()

    def test_source_output_overlap_rejected_and_default_is_sibling(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for output in (self.path/'source',self.path/'source/nested',self.path):
                with self.assertRaises(SystemExit):
                    regroup.parse_args(['--source-dir',str(self.path/'source'),'--output-dir',str(output)])
        args = regroup.parse_args(['--source-dir',str(self.path/'source')])
        self.assertEqual(args.output_dir,self.path/'source_psr_v1')

    def test_parser_and_module_are_CPU_only_no_torch_import(self):
        code = "import sys; from mizar_ablation_regroup.regroup import main; assert main(['--check-args'])==0; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        run = subprocess.run([sys.executable,'-B','-c',code],cwd=regroup.ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') is not None,'requires Matplotlib for real PNG integration')
    def test_actual_CLI_outputs29rows_PNG_SVG_and_source_bytes_unchanged(self):
        source = self.path/'source'; make_source(source)
        snapshot = {p.relative_to(source).as_posix():storage.file_sha(p) for p in source.rglob('*') if p.is_file()}
        code = "import sys; from mizar_ablation_regroup.regroup import main; code=main(sys.argv[1:]); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; raise SystemExit(code)"
        run = subprocess.run([sys.executable,'-B','-c',code,'--source-dir',str(source)],cwd=regroup.ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        output = self.path/'source_psr_v1'
        report = storage.read_json(output/'regroup_report.json')
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['group_totals'],{'perception':331,'semantic':230,'reasoning':439})
        self.assertEqual(len((output/'psr_scores.csv').read_text().splitlines()),30)
        self.assertEqual(len((output/'subcategory_mapping.csv').read_text().splitlines()),28)
        self.assertEqual(len(storage.read_json(output/'psr_scores.json')),29)
        self.assertEqual((output/'psr_accuracy_by_layer.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        from PIL import Image
        with Image.open(output/'psr_accuracy_by_layer.png') as img:
            self.assertEqual(img.size,(3000,1100))
        self.assertIn('<svg',(output/'psr_accuracy_by_layer.svg').read_text())
        after = {p.relative_to(source).as_posix():storage.file_sha(p) for p in source.rglob('*') if p.is_file()}
        self.assertEqual(snapshot,after)
        # Re-running the same offline analysis is safe with the same identity.
        with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(regroup.main(['--source-dir',str(source)]),0)


if __name__ == '__main__':
    unittest.main()
