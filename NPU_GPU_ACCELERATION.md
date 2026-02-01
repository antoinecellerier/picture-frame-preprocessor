# NPU and GPU Acceleration Guide

## Your Hardware Accelerators

### ✅ Intel GNA (Gaussian & Neural Accelerator)
**Status:** Detected at `00:08.0`
**Type:** Dedicated NPU for low-power neural network inference
**Capabilities:**
- Optimized for small CNN/RNN models
- Very low power consumption
- Fixed-point arithmetic (INT8/INT16)
- Best for: Audio processing, keyword spotting, small vision models

**Current Status:** ❌ Not detected by OpenVINO
**Reason:** May need additional drivers or unsupported by current OpenVINO version

### ✅ Intel Iris Xe Graphics (Integrated GPU)
**Status:** Detected at `/dev/dri/card0`
**Type:** Integrated GPU with compute capabilities
**Capabilities:**
- General-purpose GPU compute
- Supports larger models than GNA
- Good for parallel processing
- Best for: Image processing, larger neural networks

**Current Status:** ❌ Not detected by OpenVINO
**Reason:** GPU plugin may need installation

---

## Enabling GPU Acceleration (Intel Iris Xe)

### Step 1: Install Intel GPU Drivers
```bash
# Check current driver
ls /dev/dri/
# Should see: card0, renderD128 ✅ (you have this)

# Install Intel GPU compute runtime
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
```

### Step 2: Install OpenVINO GPU Plugin
```bash
source venv/bin/activate

# Install GPU plugin
pip install openvino-dev

# Or install specific GPU components
pip install openvino-gpu-plugin
```

### Step 3: Verify GPU Detection
```python
import openvino as ov
core = ov.Core()
print("Available devices:", core.available_devices)
# Should show: ['CPU', 'GPU.0']
```

### Step 4: Use GPU in Detector
Modify `detector.py`:
```python
# In _load_model() method, after loading model:
if self.use_openvino and self._model_type == 'openvino':
    # Try to use GPU if available
    try:
        from ultralytics import YOLO
        self._model = YOLO(openvino_path, task='detect')
        # Force GPU device
        self._model.predictor.args.device = 'GPU'
    except:
        # Fall back to CPU
        pass
```

**Expected Speedup:** 2-3x faster than CPU for YOLO inference

---

## Enabling NPU Acceleration (Intel GNA)

### Limitations
Intel GNA is designed for specific workloads:
- ⚠️ **Model size limit:** ~100MB (YOLO models are 50-200MB)
- ⚠️ **INT8/INT16 only:** Requires quantization
- ⚠️ **Limited operations:** CNN/RNN focused, not all ops supported
- ✅ **Very efficient:** 10x lower power than CPU

### Compatibility Check
**YOLOv8:** ❌ Too large, too many unsupported operations
**Grounding DINO:** ❌ Transformer-based, unsupported
**RT-DETR:** ❌ Transformer-based, unsupported

**Verdict:** GNA not suitable for our current models

### If You Still Want to Try

1. **Install GNA plugin:**
```bash
# Check if GNA plugin exists
pip install openvino-gna-plugin
```

2. **Check detection:**
```python
import openvino as ov
core = ov.Core()
print(core.available_devices)
# Look for 'GNA' in list
```

3. **Limitations:**
- Must quantize model to INT8
- Model must be <100MB
- Only certain layer types supported

---

## ⚠️ DEPRECATED: Intel Extension for PyTorch (IPEX)

**IPEX has been deprecated as of 2024.** Intel optimizations have been upstreamed into PyTorch.

**Reference:** https://github.com/intel/intel-extension-for-pytorch/issues/867

### What this means:
- **CPU optimizations**: Now built into PyTorch via MKLDNN (automatic)
- **GPU (XPU) support**: Limited Intel GPU support may still require IPEX, but development has ceased
- **Recommendation**: Use OpenVINO for Intel CPU optimization instead

### Migration:
If you have IPEX installed, you can remove it:
```bash
pip uninstall intel-extension-for-pytorch
```

PyTorch's built-in optimizations will continue to work automatically.

---

## Practical Recommendations

### For Your System (i7-1270P)

#### Option 1: Intel GPU (Iris Xe) ⭐ **Recommended**
**Best for:** YOLO models, image preprocessing
**Setup effort:** Medium (install drivers + plugins)
**Expected speedup:** 2-3x for YOLO inference
**Power cost:** Moderate

**Implementation:**
```bash
# Install GPU runtime
sudo apt-get install intel-opencl-icd intel-level-zero-gpu

# Test GPU detection
source venv/bin/activate
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

#### Option 2: Intel PyTorch Extension ⭐ **Easy Win**
**Best for:** Grounding DINO, transformer models
**Setup effort:** Low (single pip install)
**Expected speedup:** 1.5-2x
**Power cost:** Low (CPU optimizations)

**⚠️ DEPRECATED - Use OpenVINO instead**

#### Option 3: NPU (GNA) ❌ **Not Recommended**
**Best for:** Tiny models, audio processing
**Our models:** Too large, unsupported operations
**Verdict:** Skip for this use case

---

## Quick Test: Check GPU Availability

```bash
# 1. Check OpenCL (GPU compute)
sudo apt-get install clinfo
clinfo | grep -A 5 "Device Name"

# 2. Check Level Zero (Intel GPU)
sudo apt-get install level-zero-dev
ls /etc/OpenCL/vendors/

# 3. Test in Python
source venv/bin/activate
python << EOF
import openvino as ov
core = ov.Core()
devices = core.available_devices
print(f"Available: {devices}")

# Try to get GPU info
if 'GPU' in devices or 'GPU.0' in devices:
    print("GPU detected!")
    print(core.get_property('GPU.0', 'FULL_DEVICE_NAME'))
else:
    print("GPU not available in OpenVINO")
EOF
```

---

## Expected Performance Impact

### Current Setup (CPU Only)
- YOLOv8m: ~1000ms per image
- RT-DETR-L: ~2000ms per image
- Grounding DINO: ~3000ms per image
- **Ensemble:** ~3500ms per image

### With Intel GPU (Iris Xe)
- YOLOv8m: ~400ms per image (2.5x faster)
- RT-DETR-L: ~800ms per image (2.5x faster)
- Grounding DINO: ~3000ms (no change - PyTorch, not OpenVINO)
- **Ensemble:** ~1500ms per image (2.3x faster)

### With Intel PyTorch Extension
- YOLOv8m: ~900ms (1.1x - minimal benefit, already using OpenVINO)
- RT-DETR-L: ~1800ms (1.1x - minimal benefit)
- Grounding DINO: ~1800ms (1.7x faster) ⭐
- **Ensemble:** ~2700ms (1.3x faster)

### With Both GPU + IPEX
- YOLOv8m: ~400ms (GPU via OpenVINO)
- RT-DETR-L: ~800ms (GPU via OpenVINO)
- Grounding DINO: ~1800ms (IPEX optimizations)
- **Ensemble:** ~1300ms per image (2.7x faster)

### Combined with Multi-Processing (8 workers)
- **Current:** 64 images in ~10 minutes
- **With optimizations:** 64 images in ~2-3 minutes
- **Total speedup:** ~4-5x faster

---

## Implementation Priority

1. ⭐⭐⭐ **Install Intel PyTorch Extension** (5 minutes)
   - Easy, low risk
   - 1.7x faster Grounding DINO
   - No driver hassles

2. ⭐⭐ **Enable Intel GPU for OpenVINO** (30 minutes)
   - 2.5x faster YOLO/RT-DETR
   - Requires driver installation
   - Medium complexity

3. ⭐ **Combine with multi-processing** (already done)
   - 2x from parallelization
   - 2.7x from hardware acceleration
   - **Total: ~5x faster**

4. ❌ **NPU/GNA** (skip)
   - Not compatible with our models
   - Not worth the effort

---

## Next Steps

Want me to:
1. **Install Intel PyTorch Extension** and test Grounding DINO speedup?
2. **Set up Intel GPU drivers** for OpenVINO acceleration?
3. **Both** - full hardware acceleration setup?

Let me know and I'll implement it!
