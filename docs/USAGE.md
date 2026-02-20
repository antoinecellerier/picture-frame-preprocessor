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
- Contextual zoom: only zooms when subject is small relative to frame
- Falls back to saliency detection when no detections found

### Saliency

Uses OpenCV saliency detection to find visually interesting regions. Automatic fallback when smart strategy finds no detections. Applies conservative 1.2x zoom.

### Center

Simple center crop. Last resort fallback.

## How It Works

1. **Load** - Opens image, validates format, applies EXIF orientation
2. **Aspect ratio** - Determines if image needs cropping (landscape: crop width, portrait: crop height)
3. **Detect** - Runs YOLO-World + Grounding DINO ensemble to find art subjects
4. **Crop** - Centers crop window on primary subject (center-weighted, class-prioritized)
5. **Zoom** - Analyzes subject size, zooms only if needed
6. **Resize** - Scales to exact target dimensions (480x800)
7. **Save** - Exports JPEG with preserved EXIF metadata and ML analysis JSON

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
