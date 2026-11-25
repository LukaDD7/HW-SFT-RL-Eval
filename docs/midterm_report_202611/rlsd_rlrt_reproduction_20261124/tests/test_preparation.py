"""Offline checks for the new recipe and data export, using tiny image fixtures."""
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preparation', ROOT / 'prepare_hf_data.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1kAAAAASUVORK5CYII=')


def fixture():
    row = json.loads((ROOT / 'evidence/MOPD-route-v2-train-15k_first_row_metadata.json').read_text(encoding='utf-8'))
    row['image'] = {'bytes': PNG, 'path': None}
    row['image_sha256'] = hashlib.sha256(PNG).hexdigest()
    return row


class PreparationTests(unittest.TestCase):
    def test_real_metadata_mapping_keeps_answer_and_rules_separate(self):
        row = fixture()
        converted = prep.convert_row(row, 'mopd15k', 0, '/tmp/image.img')
        self.assertEqual(converted['problem'], row['question'])
        self.assertEqual(converted['answer'], 'D')
        self.assertEqual(converted['rule_solution']['references'], ['D'])
        self.assertEqual(converted['rule_solution_json'], row['rule_solution_json'])
        self.assertNotIn('prompt', converted)
        self.assertEqual(converted['published_prompt'], row['prompt'])
        self.assertNotIn('teacher_id', converted)
        self.assertEqual(converted['published_teacher_id'], row['teacher_id'])
        self.assertEqual(converted['problem_type'], 'multiple choice')
        self.assertNotIn('options', converted)  # no duplicated options in student prompt
        self.assertEqual(converted['rule_solution']['options'], ['Five', 'seven', 'three', 'two'])

    def test_duplicate_samples_remain_separate_training_rows(self):
        row = fixture()
        a = prep.convert_row(row, 'mopd15k', 0, 'image.img')
        b = prep.convert_row(row, 'mopd15k', 1, 'image.img')
        c = prep.convert_row(row, 'rstg15k', 0, 'image.img')
        self.assertEqual(len({a['problem_id'], b['problem_id'], c['problem_id']}), 3)
        self.assertEqual(a['source_sample_id'], b['source_sample_id'])

    def test_invalid_structure_fails_instead_of_silently_dropping_rows(self):
        for changes in [{'ground_truth': ''}, {'question': 'no image'}, {'rule_solution_json': '[]'}]:
            row = fixture()
            row.update(changes)
            with self.assertRaises(ValueError):
                prep.convert_row(row, 'mopd15k', 0, 'image.img')

    def test_pinned_recipes_have_expected_counts_and_same_non_data_parameters(self):
        recipe = json.loads((ROOT / 'configs/experiment_recipe.json').read_text(encoding='utf-8'))
        self.assertEqual(recipe['experiments']['base_mix30k']['datasets'], ['mopd15k', 'rstg15k'])
        self.assertEqual(recipe['experiments']['mopd_rstg15k']['datasets'], ['rstg15k'])
        self.assertFalse(recipe['data_policy']['deduplicate'])
        original = json.loads((ROOT / 'evidence/HISTORICAL_BASE_PREFLIGHT_NOT_MOPD.json').read_text(encoding='utf-8'))
        for key, value in recipe['common'].items():
            if '.' in key:
                actual = original
                for part in key.split('.'):
                    actual = actual[part]
                self.assertEqual(actual, value, key)
        for data in recipe['datasets'].values():
            manifest = json.loads((ROOT / 'evidence' / (data['repo_id'].split('/')[-1] + '_manifest.json')).read_text(encoding='utf-8'))
            self.assertEqual(sum(x['rows'] for x in manifest['shards']), 15000)
            self.assertEqual(manifest['rows'], data['rows'])

    def test_offline_parquet_export_merge_hashes_and_no_val_holdout(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'evidence').mkdir()
            (root / 'configs').mkdir()
            data = {}
            for key in ['mopd15k', 'rstg15k']:
                snap = root / 'snapshots' / key
                (snap / 'data').mkdir(parents=True)
                path = snap / 'data/train-00000-of-00001.parquet'
                rows = [deepcopy(fixture()), deepcopy(fixture())]
                pq.write_table(pa.Table.from_pylist(rows), path)
                declared = {'repo_id': 'example/' + key, 'revision': 'fixture', 'rows': 2}
                manifest = {'rows': 2, 'shards': [{'path': path.relative_to(snap).as_posix(),
                            'rows': 2, 'bytes': path.stat().st_size, 'sha256': prep.sha256(path)}]}
                prep.write_json(snap / 'manifest.json', manifest)
                prep.write_json(root / 'evidence' / (key + '_manifest.json'), manifest)
                data[key] = declared
            prep.write_json(root / 'configs/experiment_recipe.json', {'datasets': data})
            args = ['prepare_hf_data.py', '--out', str(root / 'out'), '--snapshots', str(root / 'snapshots')]
            with patch.object(prep, 'ROOT', root), patch('sys.argv', args):
                prep.main()
                with self.assertRaises(FileExistsError):
                    prep.main()
            out = root / 'out'
            report = json.loads((out / 'data_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(report['record_counts'], {'base_mix30k': 4, 'mopd_rstg15k': 2})
            self.assertEqual(report['cross_set_exact_image_question_answer_intersection'], 1)
            self.assertFalse(report['deduplicated_training_rows'])
            self.assertFalse(report['offline_shuffle'])
            self.assertEqual(len(list((out / 'images').rglob('*.img'))), 1)
            base = [json.loads(x) for x in (out / 'base_mix30k.train.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len({x['problem_id'] for x in base}), 4)
            self.assertEqual([x['source_dataset_key'] for x in base], ['mopd15k'] * 2 + ['rstg15k'] * 2)
            self.assertFalse((out / 'val.jsonl').exists())
            self.assertEqual(prep.sha256(out / 'base_mix30k.train.jsonl'), report['jsonl_sha256']['base_mix30k.train.jsonl'])
            with patch.object(prep, 'ROOT', root):
                path.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    prep.export_set(snap, 'rstg15k', data['rstg15k'], out)


if __name__ == '__main__':
    unittest.main()
