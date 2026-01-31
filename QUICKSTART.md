# Quick Start Guide

## Installation

```bash
# Install package
pip install -e .

# Download ML models (optional, for smart strategy)
python scripts/download_models.py
```

## Basic Usage

### Single Image

```bash
# Recommended: Smart ML detection with contextual zoom (default, best quality)
frame-prep process -i input.jpg -o output_dir/ -v

# Customize zoom (1.0 = no zoom, 2.0 = max zoom)
frame-prep process -i input.jpg -o output_dir/ --zoom 1.5 -v

# Use even better model for maximum quality
frame-prep process -i input.jpg -o output_dir/ --model yolov8l -v
```

### Batch Processing

```bash
# Process entire directory (uses yolov8m, 0.15 confidence, 1.3x zoom)
python scripts/batch_process.py \
  -i input_dir/ \
  -o output_dir/ \
  --workers 4

# Customize for maximum quality
python scripts/batch_process.py \
  -i input_dir/ \
  -o output_dir/ \
  --model yolov8l \
  --zoom 1.5 \
  --workers 2

# Skip already processed images
python scripts/batch_process.py \
  -i input_dir/ \
  -o output_dir/ \
  --skip-existing \
  --workers 4
```

## Cropping Strategies

- **smart** (default): YOLOv8m ML + contextual zoom - **recommended for quality**
  - Detects subjects with yolov8m (better than yolov8n)
  - Applies contextual zoom (only zooms small subjects)
  - Falls back to saliency when no detections
- **saliency** (fallback): OpenCV saliency detection with conservative zoom
- **center** (fallback): Simple center crop (last resort)

**Note:** Smart is the default with optimized settings. Just run without flags.

## Common Workflows

### OneDrive → E-ink Frame

```bash
# 1. Download from OneDrive (separate tool)
cd ../onedrive-album-downloader
python download.py --album "Art" --output ~/raw_images/

# 2. Preprocess for e-ink display (uses smart strategy by default)
cd ../picture-frame-preprocessor
python scripts/batch_process.py \
  -i ~/raw_images/ \
  -o ~/processed_images/ \
  --workers 4

# 3. Images ready for librespot-epd-nowplaying
```

## Troubleshooting

**"Failed to load YOLO model"**
```bash
python scripts/download_models.py
```

**Adjust settings if needed**
```bash
# Reduce workers for memory-constrained systems
python scripts/batch_process.py -i input/ -o output/ --workers 2

# Use faster model if speed is critical
python scripts/batch_process.py -i input/ -o output/ --model yolov8s

# Disable zoom for faster processing
python scripts/batch_process.py -i input/ -o output/ --zoom 1.0
```

**Need different dimensions**
```bash
# Any dimensions work
frame-prep process -i input.jpg -o output/ -w 600 -h 1024
```

## Testing

```bash
# Run tests
pytest tests/ -v

# Format code
black src/ scripts/ tests/
```
