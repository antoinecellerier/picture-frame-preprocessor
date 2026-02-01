# Hardware Optimization Guide

## Your System Specs
- **CPU:** Intel i7-1270P (12 cores, 16 threads)
- **RAM:** 30GB total, ~21GB available
- **GPU:** Intel Iris Xe (integrated - limited ML support)
- **Storage:** SSD (assumed)

---

## Current Performance Bottlenecks

### 1. Single-Threaded Model Inference
**Problem:** Grounding DINO and YOLO run one image at a time
**Impact:** Only using 1/16 CPU threads (6% utilization)
**Solution:** Parallel batch processing

### 2. CPU-Only Inference
**Problem:** No NVIDIA GPU for CUDA acceleration
**Impact:** 5-10x slower than GPU inference
**Current Mitigation:** OpenVINO already being used for YOLO ✅

### 3. Model Loading Overhead
**Problem:** Each worker loads model separately in parallel processing
**Impact:** High memory usage, slow startup
**Solution:** Shared memory model or sequential processing

---

## Optimization Strategies

### 🚀 Strategy 1: Multi-Process Batch Processing (Easiest)
**Current:** 4 workers in batch_process.py
**Optimal:** 8-12 workers for your 16-thread CPU

**Implementation:**
```bash
# Current (4 workers)
python scripts/batch_process.py --workers 4 --ensemble

# Optimized (10 workers - leaves headroom for system)
python scripts/batch_process.py --workers 10 --ensemble
```

**Expected Speedup:** 2-2.5x faster
**Tradeoff:** Higher memory usage (each worker loads models)
**Recommended for:** Batch processing many images

---

### ⚡ Strategy 2: OpenVINO Thread Optimization
**Current:** OpenVINO auto-selects threads
**Optimal:** Explicitly set thread count

**Implementation:**
Add to detector initialization:
```python
import openvino.runtime as ov

# Set OpenVINO to use more threads
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENVINO_INFERENCE_NUM_THREADS'] = '8'
```

**Expected Speedup:** 1.3-1.5x per image
**Tradeoff:** Less effective with multi-process
**Recommended for:** Single-image processing or sequential batches

---

### 🧠 Strategy 3: Batch Inference (Advanced)
**Current:** Process images one at a time
**Optimal:** Group images into batches for GPU-style processing

**Implementation:**
```python
# Instead of:
for image in images:
    detections = model.detect(image)

# Use:
batch_size = 4
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    batch_detections = model.detect_batch(batch)
```

**Expected Speedup:** 1.5-2x
**Tradeoff:** Requires model API changes, more complex
**Note:** YOLO supports this, Grounding DINO may need modification

---

### 💾 Strategy 4: Aggressive Caching
**Current:** Cache model detections after running
**Optimal:** Cache intermediate results too

**What to Cache:**
1. ✅ Model detections (already done)
2. ⏭️ Preprocessed images (resized, normalized)
3. ⏭️ Feature maps (if repeatedly evaluating)

**Expected Speedup:** Instant re-evaluation (<1s for 64 images)
**Tradeoff:** Disk space (~100MB per model)
**Status:** Detection caching implemented ✅

---

### 🔧 Strategy 5: Intel-Specific Optimizations
**Your CPU:** 12th Gen Intel with AVX-512 support
**Opportunities:**
1. Use Intel MKL (Math Kernel Library)
2. Enable AVX-512 in PyTorch
3. Use Intel Extension for PyTorch

**⚠️ DEPRECATED:** IPEX is no longer maintained. Intel CPU optimizations are now built into PyTorch.

**Implementation:**
```bash
# No installation needed - PyTorch includes Intel optimizations (MKLDNN) by default
```

**Reference:** https://github.com/intel/intel-extension-for-pytorch/issues/867
**Recommendation:** Use OpenVINO for maximum Intel CPU performance

---

### 📊 Strategy 6: Memory-Mapped Image Loading
**Current:** PIL loads entire image to RAM
**Optimal:** Memory-map large images

**Implementation:**
```python
from PIL import Image
import numpy as np

# Memory-efficient loading for large images
img = Image.open(path)
img.load()  # Force load to check format
if img.size[0] * img.size[1] > 10_000_000:  # >10MP
    # Process as memory-mapped array
    img_array = np.asarray(img)  # No copy
else:
    img_array = np.array(img)  # Copy to RAM
```

**Expected Speedup:** Minimal, but saves RAM
**Tradeoff:** Slightly more complex
**Recommended:** Only if hitting RAM limits

---

## Recommended Configuration by Use Case

### Use Case 1: One-Time Batch Processing (Your Current Task)
**Goal:** Process 64 images as fast as possible once

**Optimal Config:**
```bash
# Use 10 workers, leverage all CPU cores
python scripts/batch_process.py \
  --input-dir test_real_images/input \
  --output-dir test_real_images/output_ensemble \
  --ensemble \
  --workers 10
```

**Expected Time:** ~5-7 minutes for 64 images
**Current Time:** ~10-12 minutes (4 workers)
**Speedup:** ~1.7-2x

---

### Use Case 2: Repeated Evaluation (Parameter Tuning)
**Goal:** Test many configurations quickly

**Optimal Config:**
```bash
# Cache once (slow)
python scripts/cache_model_detections.py \
  --models yolov8m rtdetr-l grounding_dino

# Then evaluate instantly (fast)
python scripts/optimize_accuracy.py \
  --models yolov8m rtdetr-l grounding_dino
```

**Expected Time:** Cache once (~15 min), evaluate each config (<1 sec)
**Status:** Already implemented ✅

---

### Use Case 3: Real-Time Processing (Future)
**Goal:** Process images as they arrive

**Optimal Config:**
```python
# Single worker, optimized threading
detector = EnsembleDetector(
    models=['yolov8m', 'rtdetr-l'],
    use_openvino=True
)

# Set OpenVINO threads
os.environ['OMP_NUM_THREADS'] = '8'

# Process
result = preprocessor.process_image(image_path, output_path)
```

**Expected Time:** ~3-4 seconds per image
**Best For:** OneDrive sync workflow

---

## Quick Wins (Implement Now)

### 1. Increase Batch Workers
```bash
# Change from 4 to 10 workers
sed -i 's/default=4/default=10/' scripts/batch_process.py
```
**Speedup:** 2x
**Effort:** 1 minute

### 2. Enable OpenVINO Threading
Add to `src/frame_prep/detector.py` after imports:
```python
import os
os.environ['OMP_NUM_THREADS'] = '8'
```
**Speedup:** 1.3x (per worker)
**Effort:** 2 minutes

### 3. Use All Available RAM
Current memory usage: ~10GB / 30GB (33%)
Can safely run 12-14 workers without issues

**Combined Speedup:** ~2.6x faster overall

---

## Advanced Optimizations (Future)

### 1. Quantization (INT8 Inference)
Convert models to 8-bit integers instead of 32-bit floats
- **Speedup:** 2-4x faster
- **Tradeoff:** Slight accuracy loss (~1-2%)
- **Effort:** High (requires model conversion)

### 2. ONNX Runtime
Convert PyTorch models to ONNX for optimized inference
- **Speedup:** 1.5-2x faster
- **Tradeoff:** One-time conversion effort
- **Best for:** Production deployment

### 3. TensorRT (If NVIDIA GPU Added)
If you add an NVIDIA GPU later:
- **Speedup:** 10-20x faster than CPU
- **Cost:** ~$200-500 for RTX 3060
- **Recommended:** Only if processing >1000 images regularly

---

## Monitoring Performance

### Check CPU Utilization
```bash
# While processing, run:
htop
# Or:
top -H -p $(pgrep -f cache_grounding)
```

### Benchmark Different Configs
```bash
# Test current config
time python scripts/batch_process.py --workers 4 --ensemble

# Test optimized config
time python scripts/batch_process.py --workers 10 --ensemble
```

---

## Current Status

### Grounding DINO (Running Now)
- **Process:** Single-threaded
- **CPU Usage:** ~6% (1/16 threads)
- **Memory:** 2GB
- **Bottleneck:** Sequential processing
- **Fix:** Could parallelize with multiprocessing

### YOLOv8m + RT-DETR Ensemble
- ✅ Already using OpenVINO acceleration
- ✅ Multi-process batch processing available
- ⏭️ Could optimize thread count
- ⏭️ Could increase worker count

---

## Recommendation

**For your current Grounding DINO task:**
1. Let it finish (already 63 minutes in, probably ~50% done)
2. For next run, implement parallel version

**For future ensemble processing:**
1. Increase workers from 4 to 10
2. Add OpenVINO threading optimization
3. Expected: 64 images in ~4-5 minutes (vs current ~10 min)

**Implementation Priority:**
1. ⭐ Multi-process optimization (2x speedup, 5 min effort)
2. ⭐ OpenVINO threading (1.3x speedup, 2 min effort)
3. ⚡ Intel PyTorch extension (1.5x speedup, 15 min effort)
4. 💡 Batch inference (2x speedup, 2 hour effort)

Total potential speedup: **~5-6x faster** than current baseline
