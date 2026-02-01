#!/usr/bin/env python3
"""Check available hardware acceleration and optimizations."""

import os
import sys
from pathlib import Path


def check_cpu_info():
    """Check CPU information."""
    print("=" * 60)
    print("CPU INFORMATION")
    print("=" * 60)

    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
            model = [l for l in lines if 'model name' in l][0].split(':')[1].strip()
            cores = len([l for l in lines if l.startswith('processor')])
            print(f"Model: {model}")
            print(f"Logical cores: {cores}")
    except Exception as e:
        print(f"Could not read CPU info: {e}")

    print()


def check_pytorch():
    """Check PyTorch configuration and optimizations."""
    print("=" * 60)
    print("PYTORCH OPTIMIZATIONS")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MKLDNN available: {torch.backends.mkldnn.is_available()}")
        print(f"OpenMP available: {torch.backends.openmp.is_available()}")
        print(f"Number of threads: {torch.get_num_threads()}")
        print(f"Number of interop threads: {torch.get_num_interop_threads()}")

        # Check environment variables
        print("\nEnvironment variables:")
        for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENVINO_INFERENCE_NUM_THREADS']:
            val = os.environ.get(var, 'not set')
            print(f"  {var}: {val}")

    except ImportError:
        print("PyTorch not installed")

    print()


def check_ipex():
    """Check Intel Extension for PyTorch - DEPRECATED."""
    print("=" * 60)
    print("INTEL EXTENSION FOR PYTORCH (IPEX)")
    print("=" * 60)
    print("⚠️  IPEX is deprecated as of 2024")
    print("  See: https://github.com/intel/intel-extension-for-pytorch/issues/867")
    print("  Optimizations have been upstreamed to PyTorch")
    print("  No action needed - PyTorch includes Intel CPU optimizations")
    print()


def check_openvino():
    """Check OpenVINO installation and models."""
    print("=" * 60)
    print("OPENVINO")
    print("=" * 60)

    try:
        import openvino
        print(f"✓ OpenVINO installed: {openvino.__version__}")

        # Check for OpenVINO models
        models_dir = Path(__file__).parent.parent / 'models'
        openvino_models = list(models_dir.glob('*_openvino_model'))

        if openvino_models:
            print(f"\n  OpenVINO models found:")
            for model_path in openvino_models:
                size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / 1024 / 1024
                print(f"    - {model_path.name} ({size_mb:.1f} MB)")
        else:
            print("\n  No OpenVINO models found in models/")
            print("  To export a model to OpenVINO:")
            print("    from ultralytics import YOLO")
            print("    model = YOLO('yolov8m.pt')")
            print("    model.export(format='openvino')")

    except ImportError:
        print("✗ OpenVINO not installed")

    print()


def check_ultralytics():
    """Check Ultralytics and model availability."""
    print("=" * 60)
    print("ULTRALYTICS / YOLO")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics installed: {ultralytics.__version__}")

        # Check model cache
        models_dir = Path(__file__).parent.parent / 'models'
        pt_models = list(models_dir.glob('*.pt'))

        if pt_models:
            print(f"\n  Models found in models/ directory:")
            for model_path in pt_models:
                size_mb = model_path.stat().st_size / 1024 / 1024
                print(f"    - {model_path.name} ({size_mb:.1f} MB)")
        else:
            print("\n  No .pt models found in models/")

    except ImportError:
        print("✗ Ultralytics not installed")

    print()


def check_transformers():
    """Check transformers library for Grounding DINO."""
    print("=" * 60)
    print("TRANSFORMERS (for Grounding DINO)")
    print("=" * 60)

    try:
        import transformers
        print(f"✓ Transformers installed: {transformers.__version__}")

        # Try loading Grounding DINO
        try:
            from transformers import AutoModelForZeroShotObjectDetection
            print("  ✓ Grounding DINO support available")
        except ImportError:
            print("  ✗ Grounding DINO support not available")

    except ImportError:
        print("✗ Transformers not installed")

    print()


def print_recommendations():
    """Print optimization recommendations."""
    print("=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    import torch

    # Check OpenVINO models
    models_dir = Path(__file__).parent.parent / 'models'
    openvino_models = list(models_dir.glob('*_openvino_model'))
    pt_models = list(models_dir.glob('*.pt'))

    for pt_model in pt_models:
        model_name = pt_model.stem
        openvino_equivalent = models_dir / f"{model_name}_openvino_model"
        if not openvino_equivalent.exists() and 'world' not in model_name.lower():
            print(f"\n✓ Export {model_name} to OpenVINO for better CPU performance:")
            print(f"    from ultralytics import YOLO")
            print(f"    model = YOLO('models/{pt_model.name}')")
            print(f"    model.export(format='openvino')")

    print("\n✓ Current optimizations enabled:")
    print(f"  - MKLDNN: {torch.backends.mkldnn.is_available()}")
    print(f"  - OpenMP: {torch.backends.openmp.is_available()}")
    print(f"  - OpenVINO: Available for YOLO models")
    print(f"  - Thread count: {torch.get_num_threads()} (good for single-process)")
    print("\n  For batch processing with multiple workers:")
    print(f"    - Consider reducing threads per worker (currently 8)")
    print(f"    - Total threads = workers × threads_per_worker")

    print()


if __name__ == '__main__':
    check_cpu_info()
    check_pytorch()
    check_ipex()
    check_openvino()
    check_ultralytics()
    check_transformers()
    print_recommendations()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Run this script anytime to check optimization status")
    print("=" * 60)
