"""Read-only CUDA environment inventory. Run inside the actual Linux training venv."""
import argparse
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = {'python': sys.version, 'python_executable': sys.executable,
              'platform': platform.platform(), 'versions': {}, 'errors': []}
    expected = {'torch': '2.8.0', 'torchvision': '0.23.0', 'torchaudio': '2.8.0',
                'transformers': '4.57.6', 'vllm': '0.11.0', 'flash-attn': '2.8.3',
                'ray': '2.53.0', 'omegaconf': '2.3.1', 'qwen-vl-utils': '0.0.14'}
    for key, wanted in expected.items():
        try:
            actual = version(key)
            report['versions'][key] = actual
            if actual.split('+')[0] != wanted:
                report['errors'].append(f'{key}: expected {wanted}, got {actual}')
        except PackageNotFoundError:
            report['errors'].append(f'{key}: missing')
    if sys.version_info[:2] != (3, 11) or platform.system() != 'Linux':
        report['errors'].append('Expected Python3.11 on Linux')
    for module in ['torchvision', 'vllm', 'flash_attn', 'transformers', 'ray', 'omegaconf', 'tensordict', 'torchdata']:
        try:
            importlib.import_module(module)
        except Exception as exc:
            report['errors'].append(f'Import {module}: {type(exc).__name__}: {exc}')
    try:
        import torch
        report['torch_cuda'] = torch.version.cuda
        report['cxx11_abi'] = torch._C._GLIBCXX_USE_CXX11_ABI
        report['gpu_count'] = torch.cuda.device_count()
        report['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            report['errors'].append('Expected 8 visible CUDA devices')
        if torch.version.cuda != '12.8':
            report['errors'].append('Expected torch CUDA build 12.8')
        # Tiny allocation/matmul checks each device; not a performance or NCCL test.
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                x = torch.ones((16, 16), device=f'cuda:{i}', dtype=torch.bfloat16)
                assert (x @ x)[0, 0].item() == 16.0
                del x
    except Exception as exc:
        report['errors'].append(f'CUDA check: {type(exc).__name__}: {exc}')
    for key, cmd in [('nvidia_smi', ['nvidia-smi']), ('pip_check', [sys.executable, '-m', 'pip', 'check'])]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            report[key] = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
            if proc.returncode:
                report['errors'].append(f'{key} failed')
        except Exception as exc:
            report['errors'].append(f'{key}: {exc}')
    report['status'] = 'FAIL' if report['errors'] else 'PASS_IMPORT_AND_TINY_CUDA_ONLY'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'errors': report['errors'], 'out': str(args.out)}, ensure_ascii=False))
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
