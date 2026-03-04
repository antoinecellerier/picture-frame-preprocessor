#!/usr/bin/env python3
"""Download and initialize detection models.

Usage:
    python scripts/download_models.py           # YOLO models only
    python scripts/download_models.py --vlm     # Also download Qwen3-VL GGUF files
    python scripts/download_models.py --all     # Both
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# GGUF files hosted on HuggingFace
_VLM_REPO = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
_VLM_FILES = [
    "Qwen3VL-2B-Instruct-Q8_0.gguf",          # ~1.8 GB — quantized model weights
    "mmproj-Qwen3VL-2B-Instruct-F16.gguf",    # ~782 MB — vision encoder projection
]
_VLM_DIR = PROJECT_ROOT / "models" / "qwen3vl"


def download_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Run: pip install -e .")
        return 1

    models = ['yolov8n', 'yolov8s', 'yolov8m']
    print("Downloading YOLOv8 models...")
    print("yolov8m (52 MB) is the default for best art detection quality.\n")

    for model_name in models:
        print(f"  Downloading {model_name}.pt...")
        try:
            YOLO(f'{model_name}.pt')
            print(f"  ✓ {model_name}.pt")
        except Exception as e:
            print(f"  ✗ {model_name}.pt: {e}")
            return 1

    return 0


def download_vlm():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Run: pip install huggingface-hub")
        return 1

    _VLM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Qwen3-VL-2B GGUF files (~2.6 GB total) to {_VLM_DIR}/")

    for filename in _VLM_FILES:
        dest = _VLM_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} (already present, {dest.stat().st_size // 1_000_000} MB)")
            continue
        print(f"  Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=_VLM_REPO,
                filename=filename,
                local_dir=str(_VLM_DIR),
            )
            size_mb = (_VLM_DIR / filename).stat().st_size // 1_000_000
            print(f"  ✓ {filename} ({size_mb} MB)")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
            return 1

    print()
    _print_llama_server_instructions()
    return 0


def _print_llama_server_instructions():
    default_bin = Path.home() / "stuff" / "llama.cpp" / "build" / "bin" / "llama-server"
    if default_bin.exists():
        print(f"  ✓ llama-server found at {default_bin}")
        return

    print("llama-server binary not found. Build it once with:")
    print()
    print("  git clone https://github.com/ggerganov/llama.cpp ~/stuff/llama.cpp")
    print("  cmake -B ~/stuff/llama.cpp/build ~/stuff/llama.cpp \\")
    print("        -DGGML_AVX2=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build ~/stuff/llama.cpp/build --target llama-server -j$(nproc)")
    print()
    print("  Set LLAMA_SERVER_BIN env var if you build to a different path.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vlm', action='store_true', help='Download Qwen3-VL GGUF files (~2.6 GB)')
    p.add_argument('--all', action='store_true', help='Download YOLO + VLM models')
    args = p.parse_args()

    do_yolo = not args.vlm or args.all
    do_vlm  = args.vlm or args.all

    if do_yolo:
        print("=== YOLO models ===")
        rc = download_yolo()
        if rc:
            return rc
        print()

    if do_vlm:
        print("=== Qwen3-VL GGUF models ===")
        rc = download_vlm()
        if rc:
            return rc
        print()

    print("=" * 60)
    if do_yolo and not do_vlm:
        print("YOLO models ready.")
        print()
        print("Tip: also download VLM models for +4% accuracy on difficult images:")
        print("  python scripts/download_models.py --vlm")
    elif do_vlm:
        print("All models ready.")
        print()
        print("Run with VLM fallback:")
        print("  frame-prep process -i photo.jpg -o out/ --vlm -v")
        print("  frame-prep batch -i ~/art/ -o ~/processed/ --vlm --skip-existing")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
