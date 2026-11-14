"""Run on the authorized GPU environment, before serving or synthesizing."""
import json
import os
import sys
import time
from pathlib import Path


def gpu_probe():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('single_cuda_device_required')
    prop = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    minimum = int(os.environ.get('SPEECH_MIN_FREE_GPU_BYTES', str(18 * 1024 ** 3)))
    if free < minimum:
        raise RuntimeError('insufficient_free_gpu_memory')
    a = torch.ones((256, 256), device='cuda', dtype=torch.float16)
    if not torch.all((a @ a) == 256).item():
        raise RuntimeError('cuda_operator_failed')
    torch.cuda.synchronize()
    return {'device': prop.name, 'uuid': str(getattr(prop, 'uuid', 'unknown')), 'capability': list(torch.cuda.get_device_capability()),
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'free_bytes_before': free, 'total_bytes': total,
            'cuda_operator': 'passed'}


def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else ''
    if mode == 'gpu':
        result = gpu_probe()
    elif mode == 'models':
        from .models import verify_models
        result = verify_models(Path(os.environ.get('SPEECH_MODEL_ROOT', '/models')))
    elif mode == 'load':
        from .worker import IndexEngine
        import torch
        result = gpu_probe()
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        engine = IndexEngine(os.environ.get('SPEECH_MODEL_ROOT', '/models'))
        engine.load()
        torch.cuda.synchronize()
        result.update({'model_loaded': True, 'synthesis_tested': False, 'load_seconds': time.monotonic() - started,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    else:
        raise SystemExit('usage: python -m speech_runtime.probe gpu|models|load')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
