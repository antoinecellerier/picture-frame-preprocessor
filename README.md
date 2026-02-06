# Picture Frame Preprocessor

Intelligent image preprocessor for e-ink picture frames. Converts art and streetart images to exact portrait format (480x800) using local ML to detect and focus on key artistic features. Handles both landscape (crop width) and portrait (crop height) images intelligently.

## Features

- **ML-Powered Smart Cropping**: Ensemble detection (YOLO-World + Grounding DINO) to detect art subjects and crop intelligently
- **OpenVINO Acceleration**: Automatic GPU/NPU acceleration on Intel hardware (1.4x faster)
- **Contextual Zoom**: Automatically zooms based on subject size to remove excessive background
- **Portrait & Landscape**: Intelligently crops both orientations to exact aspect ratio
- **Multiple Strategies**: Smart (ML), saliency-based, or center cropping
- **Multiple Detectors**: Single-model (YOLOv8), ensemble (YOLOv8m + RT-DETR-L), or optimized ensemble (YOLO-World + Grounding DINO)
- **Batch Processing**: Process entire directories with parallel workers and ML analysis caching
- **EXIF Preservation**: Maintains image metadata and handles rotation
- **Flexible Output**: Configurable dimensions and quality
- **Local Processing**: All processing runs locally, no cloud dependencies

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

```bash
# 1. Clone or navigate to the repository
cd picture-frame-preprocessor

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install package
pip install -e .

# 4. Download ML models
python scripts/download_models.py
```

## Usage

### Single Image Processing

Process a single image:

```bash
frame-prep process \
  --input /path/to/image.jpg \
  --output /path/to/output/ \
  --width 480 \
  --height 800 \
  --verbose
```

Options:
- `--input, -i`: Input image file (required)
- `--output, -o`: Output directory or file path (required)
- `--width, -w`: Target width in pixels (default: 480)
- `--height, -h`: Target height in pixels (default: 800)
- `--strategy, -s`: Cropping strategy: smart, saliency, center (default: smart)
- `--single-model`: Use single YOLOv8 model instead of default ensemble (faster, lower accuracy)
- `--ensemble`: Use YOLOv8m + RT-DETR-L ensemble instead of default
- `--model, -m`: YOLO model variant for `--single-model` mode (default: yolov8m)
- `--confidence, -c`: Detection confidence threshold (default: 0.15)
- `--no-two-pass`: Disable two-pass center-crop detection (faster, may miss small centered subjects)
- `--zoom, -z`: Contextual zoom factor (default: 1.3, range: 1.0-2.0)
- `--quality, -q`: JPEG quality 1-100 (default: 95)
- `--verbose, -v`: Verbose output

### Batch Processing

Process a directory of images:

```bash
python scripts/batch_process.py \
  --input-dir ~/images/raw/ \
  --output-dir ~/images/processed/ \
  --width 480 \
  --height 800 \
  --workers 4 \
  --skip-existing
```

Options:
- `--input-dir, -i`: Input directory (required)
- `--output-dir, -o`: Output directory (required)
- `--width, -w`: Target width (default: 480)
- `--height`: Target height (default: 800)
- `--strategy, -s`: Cropping strategy (default: smart)
- `--single-model`: Use single YOLOv8 model instead of default ensemble (faster, lower accuracy)
- `--ensemble`: Use YOLOv8m + RT-DETR-L ensemble instead of default
- `--model, -m`: YOLO model variant for `--single-model` mode (default: yolov8m)
- `--confidence, -c`: Detection threshold (default: 0.15)
- `--zoom, -z`: Contextual zoom factor (default: 1.3)
- `--workers`: Number of parallel workers (default: 8, optimized for 16-thread CPUs)
- `--threads-per-worker`: Threads per worker process (default: 4)
- `--no-two-pass`: Disable two-pass center-crop detection (faster, may miss small centered subjects)
- `--no-openvino`: Disable OpenVINO acceleration (enabled by default)
- `--skip-existing`: Skip already processed images
- `--recursive, -r`: Process subdirectories recursively

**Hardware Acceleration:**
- OpenVINO is enabled by default for 1.4-2.0x CPU speedup
- Models are cached per worker for 6.7x faster batch processing
- See `docs/HARDWARE_ACCELERATION.md` for optimization guide
- Use `python scripts/check_optimizations.py` to check system status

## Cropping Strategies

### Smart (Default - Recommended)
**Uses ML detection to detect subjects and intelligently crops with contextual zoom.** This is the default and recommended strategy for best quality results.

**Detector options:**
- Default: YOLO-World + Grounding DINO optimized ensemble (best accuracy)
- `--single-model`: Single YOLOv8m model (faster, lower accuracy)
- `--ensemble`: YOLOv8m + RT-DETR-L ensemble (moderate accuracy)

**Features:**
- Detects people, art, sculptures, murals, and other subjects
- Center-weighted primary subject selection with class priorities
- Contextual zoom: Only zooms when subject is small (removes excessive background)
- Falls back to saliency detection when no detections found
- Optimized for art: Lower threshold (0.15), art-specific class priorities

**Use this for quality.** Simply omit the `--strategy` flag:
```bash
# Default: optimized ensemble (YOLO-World + Grounding DINO)
frame-prep process -i input.jpg -o output/ -v

# Faster with single model (lower accuracy)
frame-prep process -i input.jpg -o output/ --single-model -v
```

### Saliency (Fallback)
Uses OpenCV saliency detection to find visually interesting regions (art). Automatic fallback when smart strategy finds no YOLO detections. Applies conservative 1.2x zoom.

### Center (Fallback)
Simple center crop. Last resort fallback when other strategies unavailable.

## How It Works

1. **Load Image**: Opens image, validates format, applies EXIF orientation
2. **Check Aspect Ratio**: Determines if image needs cropping to reach target 3:5 ratio
   - Landscape (too wide) → Crop width horizontally to focus on subjects
   - Portrait (too tall) → Crop height vertically to focus on subjects
3. **Detect Subjects**: Runs YOLO-World + Grounding DINO ensemble by default to find art subjects (or saliency fallback)
4. **Calculate Crop**: Centers crop window on primary subject (selected via center-weighting and class priorities)
5. **Contextual Zoom**: Analyzes subject size, zooms only if needed to focus on art
6. **Apply Crop**: Crops to exact portrait aspect ratio with zoom applied
7. **Resize**: Scales to exact target dimensions (480x800)
8. **Save**: Exports as JPEG with preserved EXIF metadata and ML analysis JSON

**Contextual Zoom Logic:**
- Large subject (>60% of frame) → No zoom
- Medium subject (40-60%) → Slight zoom (1.15x)
- Small subject (20-40%) → Moderate zoom (calculated)
- Tiny subject (<20%) → Max zoom (up to 1.3x)

See [docs/CONTEXTUAL_ZOOM.md](docs/CONTEXTUAL_ZOOM.md) for details.

## Workflow Integration

### With [OneDrive Album Downloader](https://github.com/antoinecellerier/onedrive-album-download)

```bash
# Download from OneDrive
cd ../onedrive-album-downloader
python download.py --album "Art Collection" --output ~/images/raw/

# Batch preprocess
cd ../picture-frame-preprocessor
python scripts/batch_process.py \
  --input-dir ~/images/raw/ \
  --output-dir ~/images/processed/ \
  --width 480 --height 800

# Images now ready for e-ink display
```

### With [librespot-epd-nowplaying](https://github.com/antoinecellerier/librespot-epd-nowplaying)

Processed images are ready for use with the e-ink frame's idle mode display.

## Performance

- **ML detection + crop** (yolov8m, OpenVINO CPU): ~0.5-0.8s per image
- **ML detection + crop** (yolov8m, PyTorch CPU): ~1.1-1.5s per image
- **Batch processing**: ~100-120 images/minute on CPU with yolov8m + OpenVINO (4 workers)

**OpenVINO Acceleration:**
- Automatically enabled on Intel CPUs (1.4x faster than PyTorch)
- First inference: 3x faster (1.1s → 0.3s)
- Subsequent: 1.2x faster (0.7s → 0.5s)
- No configuration needed - detects and uses OpenVINO model automatically

**Model Performance Trade-offs:**
| Model | Speed (OpenVINO) | Speed (PyTorch) | Quality | Recommended For |
|-------|------------------|-----------------|---------|-----------------|
| yolov8n | Fast (~0.2s) | Fast (~0.3s) | Good | Quick testing |
| yolov8s | Fast (~0.4s) | Fast (~0.5s) | Better | Speed priority |
| yolov8m | Medium (~0.7s) | Medium (~1.1s) | **Best** | **Quality (default)** |
| yolov8l | Slow (~1.5s) | Slow (~2.5s) | Excellent | Maximum quality |

## Troubleshooting

### "Failed to load YOLO model" Error

Run the model download script:
```bash
python scripts/download_models.py
```

### Slow Processing

- Use `--single-model` for faster processing (single YOLOv8m instead of ensemble)
- Use fewer workers (`--workers 2`)
- Use simpler strategy (`--strategy center`)

### Poor Crop Results

- The default optimized ensemble should give best results
- Try `--strategy saliency` for images where ML detection struggles
- With `--single-model`, try adjusting `--confidence 0.15`

## Project Structure

```
picture-frame-preprocessor/
├── README.md
├── requirements.txt
├── setup.py
├── src/frame_prep/
│   ├── __init__.py
│   ├── cli.py              # Click CLI entry point
│   ├── preprocessor.py     # Core pipeline orchestration
│   ├── detector.py         # Detection models (YOLOv8, Ensemble, OptimizedEnsemble)
│   ├── cropper.py          # Intelligent cropping with contextual zoom
│   ├── analyzer.py         # Saliency analysis
│   └── utils.py            # Shared utilities
├── scripts/
│   ├── batch_process.py              # Directory batch processing
│   ├── download_models.py            # Initialize models
│   ├── generate_test_set.py          # Generate random test sets
│   ├── generate_interactive_report.py # Interactive detection report (HTML)
│   ├── generate_quality_report.py    # Quality assessment report (HTML)
│   ├── check_optimizations.py        # Check system optimization status
│   └── export_to_openvino.py         # Export models to OpenVINO format
├── docs/
│   ├── TESTING_GUIDE.md              # Quality assessment guide
│   ├── CONTEXTUAL_ZOOM.md            # Zoom logic documentation
│   └── HARDWARE_ACCELERATION.md      # Hardware optimization guide
└── tests/
    └── fixtures/sample_images/
```

## Quality Assessment

Test the preprocessor on sample images and evaluate crop quality:

### Generate Test Set
```bash
python scripts/generate_test_set.py \
  --source ~/images/source/ \
  --output test_images/input \
  --count 64 \
  --seed 42
```

### Process Test Set
```bash
python scripts/batch_process.py \
  --input-dir test_images/input \
  --output-dir test_images/output
```

### Create Interactive Detection Report
```bash
python scripts/generate_interactive_report.py
```

**Features:**
- ⚙️ Configuration summary (models, parameters, thresholds)
- 🖼️ Side-by-side detection and result images
- 📏 Zoom factor displayed for each result
- ✅ Ground truth comparison with IoU scores
- 👍 Rate results: Good/Poor/Zoom Issue
- 📥 Export feedback as JSON

### Create Quality Report (Alternative)
```bash
python scripts/generate_quality_report.py \
  --input-dir test_images/input \
  --output-dir test_images/output \
  --html quality_report.html
```

See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for detailed instructions.

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ tests/
```

## Related Projects

- [onedrive-album-download](https://github.com/antoinecellerier/onedrive-album-download) - Download photo albums from OneDrive
- [librespot-epd-nowplaying](https://github.com/antoinecellerier/librespot-epd-nowplaying) - Spotify now-playing display for e-ink frames

## License

MIT

## Author

Antoine
