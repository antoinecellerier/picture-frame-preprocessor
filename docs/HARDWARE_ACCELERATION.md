# Hardware Acceleration Guide

This document explains the hardware acceleration and optimization features available in the Picture Frame Preprocessor.

## Current System

- **CPU**: 12th Gen Intel Core i7-1270P (12 cores, 16 threads)
- **PyTorch**: 2.10.0 with MKLDNN and OpenMP support (Intel optimizations built-in)
- **OpenVINO**: 2025.4.1 ✓ Installed and working

## Enabled Optimizations

### 1. OpenVINO (Primary CPU Acceleration)

**Status**: ✓ Enabled by default

OpenVINO provides 1.4-2.0x speedup for YOLO inference on Intel CPUs.

**Usage in batch processing**:
```bash
# OpenVINO is enabled by default
python scripts/batch_process.py -i input/ -o output/ --workers 8

# Disable if needed
python scripts/batch_process.py -i input/ -o output/ --no-openvino
```

**Export models to OpenVINO format**:
```bash
# Export commonly used models (yolov8m, rtdetr-l)
python scripts/export_to_openvino.py

# Export specific models
python scripts/export_to_openvino.py yolov8m rtdetr-l

# Export all models
python scripts/export_to_openvino.py --all

# Use FP16 precision (smaller, may be faster)
python scripts/export_to_openvino.py --half
```

**Currently exported models**:
- `yolov8m_openvino_model/` (99.1 MB) - Default model for art detection

### 2. MKLDNN (Intel Math Kernel Library)

**Status**: ✓ Enabled automatically by PyTorch

MKLDNN provides optimized operations for Intel CPUs (convolutions, pooling, etc.).

No configuration needed - automatically used by PyTorch.

### 3. OpenMP (Multi-threading)

**Status**: ✓ Enabled

OpenMP parallelizes operations across CPU cores.

**Thread allocation for batch processing**:
- Default: 4 threads per worker (optimal for 8 workers on 16-core CPU)
- Formula: `threads_per_worker = total_cores / num_workers`

```bash
# Default: 4 threads per worker
python scripts/batch_process.py -i input/ -o output/ --workers 8

# Custom thread count per worker
python scripts/batch_process.py -i input/ -o output/ --workers 8 --threads-per-worker 2
```

### 4. Intel CPU Optimizations (Built into PyTorch)

**Status**: ✓ Enabled automatically

Modern PyTorch includes Intel CPU optimizations that were previously provided by IPEX (Intel Extension for PyTorch).

**Note about IPEX**: IPEX has been deprecated as of 2024. Intel CPU optimizations have been upstreamed directly into PyTorch, so no additional extension is needed. See [GitHub issue #867](https://github.com/intel/intel-extension-for-pytorch/issues/867) for details.

**What you get automatically**:
- Optimized operations for Intel CPUs via MKLDNN
- Efficient threading via OpenMP
- No configuration needed - works out of the box

## Performance Tuning

### Batch Processing Workers

Adjust the number of parallel workers based on your CPU:

```bash
# Conservative (low CPU usage, good for background processing)
python scripts/batch_process.py -i input/ -o output/ --workers 4

# Balanced (recommended for 12-16 core CPUs)
python scripts/batch_process.py -i input/ -o output/ --workers 8

# Aggressive (maximum speed, high CPU usage)
python scripts/batch_process.py -i input/ -o output/ --workers 12
```

**Rule of thumb**:
- Workers = CPU cores × 0.5 to 1.0
- For 16-thread CPU: 8 workers is optimal

### Thread Allocation

Prevent over-subscription by tuning threads per worker:

```bash
# Formula: threads_per_worker × workers ≈ total_CPU_threads
# Example for 16-thread CPU with 8 workers:
python scripts/batch_process.py -i input/ -o output/ --workers 8 --threads-per-worker 2
```

### Memory Optimization

For large batches, enable skip-existing to resume interrupted processing:

```bash
python scripts/batch_process.py -i input/ -o output/ --skip-existing
```

## Checking Optimization Status

Run the optimization check script anytime:

```bash
python scripts/check_optimizations.py
```

This shows:
- CPU information
- Available optimizations (MKLDNN, OpenMP, OpenVINO)
- Installed models
- Recommendations for improvement

## Performance Comparison

| Configuration | Relative Speed | Best For |
|--------------|----------------|----------|
| PyTorch only (no OpenVINO) | 1.0x | Baseline |
| PyTorch + MKLDNN + OpenMP | 1.2x | Automatic (built-in) |
| PyTorch + OpenVINO | 1.4-2.0x | **Recommended** |

## GPU Support

Currently disabled (no NVIDIA GPU detected).

If running on a machine with GPU:
1. Install CUDA-enabled PyTorch
2. Models will automatically use GPU when available
3. Expect 3-10x speedup over CPU

## Troubleshooting

### OpenVINO model not loading

```python
# Verify model exists
ls models/yolov8m_openvino_model/

# Re-export if needed
python scripts/export_to_openvino.py yolov8m
```

### About IPEX

IPEX (Intel Extension for PyTorch) has been deprecated. Intel CPU optimizations are now built directly into PyTorch. If you see IPEX mentioned in old documentation or have it installed, you can safely uninstall it:
```bash
pip uninstall intel-extension-for-pytorch
```

### Slow batch processing

1. Check worker count: `--workers 8` (adjust for your CPU)
2. Verify OpenVINO is enabled (it is by default)
3. Ensure thread count is appropriate: `--threads-per-worker 4`
4. Use `--skip-existing` to resume interrupted batches

## Summary

**Current optimal configuration**:
```bash
python scripts/batch_process.py \
  -i input/ \
  -o output/ \
  --workers 8 \
  --threads-per-worker 4
  # OpenVINO enabled by default
```

This configuration provides the best performance on the current hardware (Intel i7-1270P) without requiring any PyTorch version changes.
