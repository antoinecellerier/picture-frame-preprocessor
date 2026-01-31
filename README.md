# Picture Frame Preprocessor

Intelligent image preprocessor for e-ink picture frames. Converts art and streetart images to exact portrait format (480x800) using local ML to detect and focus on key artistic features. Handles both landscape (crop width) and portrait (crop height) images intelligently.

## Features

- **ML-Powered Smart Cropping**: Uses YOLOv8 to detect subjects and crop intelligently
- **OpenVINO Acceleration**: Automatic GPU/NPU acceleration on Intel hardware (1.4x faster)
- **Contextual Zoom**: Automatically zooms based on subject size to remove excessive background
- **Portrait & Landscape**: Intelligently crops both orientations to exact aspect ratio
- **Multiple Strategies**: Smart (ML), saliency-based, or center cropping
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
- `--model, -m`: YOLO model variant (default: yolov8m for better art detection)
- `--confidence, -c`: Detection confidence threshold (default: 0.15 for more detections)
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
- `--model, -m`: YOLO model (default: yolov8m)
- `--confidence, -c`: Detection threshold (default: 0.15)
- `--zoom, -z`: Contextual zoom factor (default: 1.3)
- `--workers`: Number of parallel workers (default: 4)
- `--skip-existing`: Skip already processed images
- `--recursive, -r`: Process subdirectories recursively

## Cropping Strategies

### Smart (Default - Recommended)
**Uses YOLOv8m ML model to detect subjects and intelligently crops with contextual zoom.** This is the default and recommended strategy for best quality results.

**Features:**
- Detects people, objects in art/streetart scenes
- Contextual zoom: Only zooms when subject is small (removes excessive background)
- Falls back to saliency detection when YOLO finds nothing
- Optimized for art: Better model (yolov8m), lower threshold (0.15)

**Use this for quality.** Simply omit the `--strategy` flag:
```bash
frame-prep process -i input.jpg -o output/ -v
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
3. **Detect Subjects**: Runs YOLOv8m with OpenVINO acceleration to find people, objects (or saliency fallback)
4. **Calculate Crop**: Centers crop window on primary detection
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

### With OneDrive Album Downloader

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

### With librespot-epd-nowplaying

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

- Use fewer workers (`--workers 2`)
- Use faster model (`--model yolov8s` or `--model yolov8n`)
- Note: yolov8m is default for better art detection quality
- Use simpler strategy (`--strategy center`)

### Poor Crop Results

- Adjust confidence threshold (`--confidence 0.15`)
- Try different strategy (`--strategy saliency`)
- Use larger model (`--model yolov8s`)

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
│   ├── detector.py         # YOLOv8 wrapper
│   ├── cropper.py          # Intelligent cropping
│   ├── analyzer.py         # Saliency analysis
│   └── utils.py            # Shared utilities
├── scripts/
│   ├── batch_process.py        # Directory batch processing
│   ├── download_models.py      # Initialize models
│   ├── generate_test_set.py    # Generate random test sets
│   └── generate_quality_report.py  # Interactive quality assessment HTML
├── docs/
│   └── TESTING_GUIDE.md        # Quality assessment guide
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

### Create Interactive Quality Report
```bash
python scripts/generate_quality_report.py \
  --input-dir test_images/input \
  --output-dir test_images/output \
  --html quality_report.html
```

**Features:**
- ⭐ Rate crops 1-5 stars
- 💬 Add comments
- 📊 Live statistics
- 💾 Export feedback as JSON
- 🔄 Auto-saves progress

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

## License

MIT

## Author

Antoine
