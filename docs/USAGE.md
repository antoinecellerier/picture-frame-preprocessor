# Usage Reference

Full CLI reference for `frame-prep`.

## Single Image Processing

```bash
frame-prep process \
  --input /path/to/image.jpg \
  --output /path/to/output/ \
  --verbose
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | required | Input image file |
| `--output` | `-o` | required | Output directory or file path |
| `--width` | `-w` | 480 | Target width in pixels |
| `--height` | `-h` | 800 | Target height in pixels |
| `--strategy` | `-s` | smart | Cropping strategy: `smart`, `saliency`, `center` |
| `--single-model` | | | Use single YOLOv8 model (faster, lower accuracy) |
| `--ensemble` | | | Use YOLOv8m + RT-DETR-L ensemble |
| `--model` | `-m` | yolov8m | YOLO model variant for `--single-model` mode |
| `--confidence` | `-c` | 0.25 | Detection confidence threshold |
| `--no-two-pass` | | | Disable two-pass center-crop detection |
| `--zoom` | `-z` | 8.0 | Max contextual zoom factor |
| `--quality` | `-q` | 95 | JPEG quality 1-100 |
| `--vlm` | | | Enable Qwen3-VL fallback (see [VLM section](#vlm-fallback)) |
| `--vlm-confirm` | | | Run VLM on every image (implies `--vlm`; slow first run) |
| `--vlm-gguf` | | models/qwen3vl/…Q8_0.gguf | Path to Qwen3-VL GGUF model |
| `--vlm-mmproj` | | models/qwen3vl/…mmproj.gguf | Path to mmproj GGUF file |
| `--vlm-max-image-size` | | 512 | Max image dimension for VLM inference |
| `--verbose` | `-v` | | Verbose output |

## Batch Processing

```bash
frame-prep batch \
  --input-dir ~/images/raw/ \
  --output-dir ~/images/processed/ \
  --workers 4 \
  --skip-existing
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input-dir` | `-i` | required | Input directory |
| `--output-dir` | `-o` | required | Output directory |
| `--width` | `-w` | 480 | Target width in pixels |
| `--height` | `-h` | 800 | Target height in pixels |
| `--strategy` | `-s` | smart | Cropping strategy |
| `--single-model` | | | Use single YOLOv8 model |
| `--ensemble` | | | Use YOLOv8m + RT-DETR-L ensemble |
| `--model` | `-m` | yolov8m | YOLO model variant |
| `--confidence` | `-c` | 0.25 | Detection threshold |
| `--zoom` | `-z` | 8.0 | Max contextual zoom factor |
| `--workers` | | 8 | Parallel workers |
| `--threads-per-worker` | | 4 | Threads per worker process |
| `--no-two-pass` | | | Disable two-pass detection |
| `--no-openvino` | | | Disable OpenVINO acceleration |
| `--skip-existing` | | | Skip already processed images |
| `--recursive` | `-r` | | Process subdirectories recursively |
| `--since` | | | Only process files modified on/after this date (`YYYY-MM-DD` or ISO datetime; batch only) |
| `--until` | | | Only process files modified on/before this date (date-only is inclusive of the whole day; batch only) |
| `--vlm` | | | Enable Qwen3-VL fallback (see [VLM section](#vlm-fallback)) |
| `--vlm-confirm` | | | Run VLM on every image (implies `--vlm`) |
| `--vlm-gguf` | | models/qwen3vl/…Q8_0.gguf | Path to Qwen3-VL GGUF model |
| `--vlm-mmproj` | | models/qwen3vl/…mmproj.gguf | Path to mmproj GGUF file |
| `--vlm-max-image-size` | | 512 | Max image dimension for VLM inference |

## Cropping Strategies

### Smart (Default)

Uses ML ensemble detection (YOLO-World + Grounding DINO) to find art subjects and crop intelligently.

**Detector options:**
- Default: YOLO-World + Grounding DINO optimized ensemble (best accuracy)
- `--single-model`: Single YOLOv8m model (faster, lower accuracy)
- `--ensemble`: YOLOv8m + RT-DETR-L ensemble (moderate accuracy)

**Features:**
- Detects people, art, sculptures, murals, and other subjects
- Center-weighted primary subject selection with class priorities
- Focal point detection: when the primary fills the frame, a second Grounding DINO pass searches for faces/figures within the primary to use as the crop anchor (skipped for 3D art)
- Contextual zoom: only zooms when subject is small relative to frame
- Falls back to saliency detection when no detections found

### Saliency

Uses OpenCV saliency detection to find visually interesting regions. Automatic fallback when smart strategy finds no detections. Applies conservative 1.2x zoom.

### Center

Simple center crop. Last resort fallback.

## How It Works

1. **Load** - Opens image, validates format, applies EXIF orientation
2. **Aspect ratio** - Determines if image needs cropping (landscape: crop width, portrait: crop height)
3. **Detect** - Main pass: YOLO-World + Grounding DINO ensemble finds art subjects and selects primary
4. **Focal pass** - If primary fills the frame (zoom would be ≤1.0), runs Grounding DINO with face/figure prompts on the primary's zone to find a crop anchor inside large murals; skipped for 3D art (sculptures, statues)
5. **Crop** - Centers crop window on primary (or focal anchor if found), clamped to keep primary fully visible
6. **Zoom** - Analyzes subject size, zooms only if needed
7. **Resize** - Scales to exact target dimensions (480x800)
8. **Save** - Exports JPEG with preserved EXIF metadata

**Contextual zoom logic:**

| Subject size | Zoom applied |
|-------------|-------------|
| Large (>60% of frame) | None |
| Medium (45-65%) | Slight (up to 1.2x) |
| Small (25-45%) | Moderate |
| Tiny (<25%) | Aggressive (up to `--zoom` cap) |

See [CONTEXTUAL_ZOOM.md](CONTEXTUAL_ZOOM.md) for details.

## Performance

| Configuration | Speed | Notes |
|--------------|-------|-------|
| Default ensemble + OpenVINO | ~0.5-0.8s/image | Recommended |
| Default ensemble + PyTorch | ~1.1-1.5s/image | No OpenVINO |
| Batch (4 workers, OpenVINO) | ~100-120 images/min | CPU |

OpenVINO is enabled by default on Intel CPUs for 1.4-2x speedup. See [HARDWARE_ACCELERATION.md](HARDWARE_ACCELERATION.md) for tuning.

## VLM Fallback

`--vlm` adds a Qwen3-VL-2B grounding pass that activates when the YOLO/DINO ensemble is uncertain. It improves accuracy from 88% to 92% on the test set, particularly for mosaics, small sculptures, and ambiguous street art.

### When VLM fires

- **Fallback**: no viable central candidate found after both YOLO/DINO passes
- **Heuristic-C**: top two detections score within 20% of each other (coin-flip tie)
- **Heuristic-D**: primary detection confidence below 0.35 (uncertain pick)
- **Confirm mode** (`--vlm-confirm`): always, on every image

### Setup

Requires llama.cpp server binary and two GGUF files (~2.6 GB total):

```bash
# Download GGUF models (~2.6 GB total)
# Qwen3VL-2B-Instruct-Q8_0.gguf (1.8 GB)
# mmproj-Qwen3VL-2B-Instruct-F16.gguf (782 MB)
# Place both in models/qwen3vl/

# Build llama-server (one-time, needs cmake)
git clone https://github.com/ggerganov/llama.cpp ~/stuff/llama.cpp
cmake -B ~/stuff/llama.cpp/build ~/stuff/llama.cpp -DGGML_AVX2=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/stuff/llama.cpp/build --target llama-server -j$(nproc)
```

Use `LLAMA_SERVER_BIN` env var to override the binary path (default: `~/stuff/llama.cpp/build/bin/llama-server`).

### Usage

```bash
# Single image — VLM fires automatically on uncertain/missed detections
frame-prep process -i photo.jpg -o output/ --vlm -v

# Batch — server starts once, stays open for the full run
frame-prep batch -i ~/art/ -o ~/processed/ --vlm --skip-existing
```

### Performance

| Mode | Speed (first run) | Speed (cached) |
|------|-------------------|----------------|
| No VLM | ~0.8s/image | ~0.8s/image |
| `--vlm` (fires on ~25% of images) | +20s for triggered images | +0s (instant cache) |
| `--vlm-confirm` (fires on all) | ~20s/image | ~0s/image |

VLM results are cached in `cache/qwen3vl/` keyed by image path + model + prompt. Subsequent runs on the same images are instant.

## Troubleshooting

### "Failed to load YOLO model"

Run the model download script:
```bash
python scripts/download_models.py
```

### Slow Processing

- Use `--single-model` for faster processing
- Use fewer workers (`--workers 2`)
- Use simpler strategy (`--strategy center`)

### Poor Crop Results

- The default optimized ensemble gives best results
- Try `--strategy saliency` for images where ML detection struggles
- Adjust `--confidence` (lower catches more, e.g. `0.15`)

## Project Structure

```
picture-frame-preprocessor/
├── src/frame_prep/
│   ├── cli.py              # Click CLI entry point
│   ├── preprocessor.py     # Core pipeline orchestration
│   ├── detector.py         # Detection models (YOLO-World, Ensemble, etc.)
│   ├── cropper.py          # Intelligent cropping with contextual zoom
│   ├── analyzer.py         # Saliency analysis
│   └── utils.py            # Shared utilities
├── scripts/
│   ├── download_models.py            # Initialize models
│   ├── generate_test_set.py          # Generate random test sets
│   ├── check_optimizations.py        # Check system optimization status
│   ├── export_to_openvino.py         # Export models to OpenVINO format
│   └── create_sample_composites.py   # Generate README sample images
├── docs/
│   ├── USAGE.md                      # This file
│   ├── TESTING_GUIDE.md              # Quality assessment guide
│   ├── CONTEXTUAL_ZOOM.md            # Zoom logic documentation
│   └── HARDWARE_ACCELERATION.md      # Hardware optimization guide
└── tests/
```
