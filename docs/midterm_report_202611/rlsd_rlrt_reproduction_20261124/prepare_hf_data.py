"""Export the two pinned HF question sets; does not implement a reward verifier."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def convert_row(row, dataset_key, index, image_path):
    """Keep answers, rule metadata and published settings in separate fields."""
    rule = json.loads(row['rule_solution_json'])
    if not isinstance(rule, dict):
        raise ValueError('rule_solution_json must encode an object')
    question = row['question']
    answer = row['ground_truth']
    if not isinstance(question, str) or question.count('<image>') != 1:
        raise ValueError('Published single-image row must have exactly one <image> marker')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('An independent nonempty final answer is required')
    answer_type = str(row.get('answer_type') or rule.get('answer_type') or '').lower()
    # Existing unified.jinja names; unknown types retain the generic prompt.
    prompt_type = {
        'mcq': 'multiple choice', 'choice': 'multiple choice',
        'multiple_choice': 'multiple choice', 'multiple choice': 'multiple choice',
        'numeric': 'numerical', 'number': 'numerical', 'numerical': 'numerical',
        'boolean': 'boolean', 'bool': 'boolean', 'math': 'math',
    }.get(answer_type, 'open-ended')
    keep = {k: v for k, v in row.items() if k != 'image'}
    for key in ('prompt', 'enable_thinking', 'min_pixels', 'max_pixels', 'teacher_id'):
        if key in keep:
            keep['published_' + key] = keep.pop(key)
    keep.update(
        problem_id=f'{dataset_key}:{index:06d}:{row["sample_id"]}',
        source_sample_id=row['sample_id'], source_dataset_key=dataset_key,
        source_row_index=index, problem=question, answer=answer,
        images=[image_path], problem_type=prompt_type, rule_solution=rule,
    )
    # question already contains options; do not append them a second time to the prompt.
    # The original verifier must receive rule_solution.options/references/tolerances.
    return keep


def check_snapshot(snapshot, key, declared):
    manifest_path = snapshot / 'manifest.json'
    frozen = ROOT / 'evidence' / (declared['repo_id'].split('/')[-1] + '_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest != json.loads(frozen.read_text(encoding='utf-8')):
        raise ValueError(f'{key}: manifest differs from the pinned evidence')
    if manifest['rows'] != declared['rows']:
        raise ValueError(f'{key}: unexpected row count')
    return manifest


def export_set(snapshot, key, declared, out):
    import pyarrow.parquet as pq

    manifest = check_snapshot(snapshot, key, declared)
    records = []
    for shard in manifest['shards']:
        path = snapshot / shard['path']
        if path.stat().st_size != shard['bytes'] or sha256(path) != shard['sha256']:
            raise ValueError(f'Shard checksum mismatch: {path}')
        table = pq.ParquetFile(path)
        if table.metadata.num_rows != shard['rows']:
            raise ValueError(f'Shard row count mismatch: {path}')
        for batch in table.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                raw_image = row['image']
                raw_image = raw_image.get('bytes') if isinstance(raw_image, dict) else raw_image
                if not isinstance(raw_image, bytes) or not raw_image:
                    raise ValueError('Expected image bytes embedded in the pinned Parquet')
                digest = hashlib.sha256(raw_image).hexdigest()
                if digest != row['image_sha256']:
                    raise ValueError('Image checksum mismatch')
                image_path = out / 'images' / digest[:2] / (digest + '.img')
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if image_path.exists():
                    if sha256(image_path) != digest:
                        raise ValueError(f'Existing image bytes differ: {image_path}')
                else:
                    image_path.write_bytes(raw_image)
                # Absolute paths avoid an ambiguous train/val image root after relocation.
                records.append(convert_row(row, key, len(records), image_path.as_posix()))
        print(f'EXPORTED {key} {len(records)}/{declared["rows"]}', flush=True)
    if len(records) != declared['rows']:
        raise ValueError(f'{key}: final row count mismatch')
    return records


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path,
                        help='Optional offline parent containing mopd15k/ and rstg15k/')
    args = parser.parse_args()
    recipe = json.loads((ROOT / 'configs/experiment_recipe.json').read_text(encoding='utf-8'))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ['base_mix30k.train.jsonl', 'mopd_rstg15k.train.jsonl', 'data_manifest.json']:
        if (out / name).exists():
            raise FileExistsError(f'Use a new output directory: {out / name}')
    sets = {}
    for key, declared in recipe['datasets'].items():
        if args.snapshots:
            snapshot = args.snapshots / key
        else:
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(
                repo_id=declared['repo_id'], repo_type='dataset',
                revision=declared['revision'], max_workers=8,
                allow_patterns=['manifest.json', 'README.md', 'data/train-*.parquet'],
            ))
        sets[key] = export_set(snapshot, key, declared, out)
    base = sets['mopd15k'] + sets['rstg15k']
    mopd = sets['rstg15k']
    # No offline shuffle: the original data.shuffle=true, seed=1 owns the permutation.
    for name, rows in [('base_mix30k.train.jsonl', base), ('mopd_rstg15k.train.jsonl', mopd)]:
        if len({r['problem_id'] for r in rows}) != len(rows):
            raise ValueError('Internal problem_id collision')
        write_jsonl(out / name, rows)
    def fingerprints(rows):
        return {(r['image_sha256'], r['question'].strip(), r['ground_truth'].strip()) for r in rows}
    a, b = fingerprints(sets['mopd15k']), fingerprints(sets['rstg15k'])
    report = {
        'datasets': recipe['datasets'], 'record_counts': {'base_mix30k': len(base), 'mopd_rstg15k': len(mopd)},
        'source_counts': {'base_mix30k': dict(Counter(r['dataset'] for r in base)),
                          'mopd_rstg15k': dict(Counter(r['dataset'] for r in mopd))},
        'cross_set_exact_image_question_answer_intersection': len(a & b),
        'base_unique_image_question_answer_fingerprints': len(a | b),
        'fingerprint_scope': 'Exact image bytes + stripped question + stripped final answer; not semantic deduplication',
        'deduplicated_training_rows': False, 'offline_shuffle': False, 'training_shuffle_seed': 1,
        'published_prompt_used_for_training': False, 'verifier': 'Original OSS v3 still required; no grading performed',
        'val': 'NOT_CREATED_OR_REMOVED_FROM_TRAIN',
        'jsonl_sha256': {name: sha256(out / name) for name in ['base_mix30k.train.jsonl', 'mopd_rstg15k.train.jsonl']},
        'absolute_image_root': str(out / 'images'),
    }
    write_json(out / 'data_manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
