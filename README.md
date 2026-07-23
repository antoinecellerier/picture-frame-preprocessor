# Picture Frame Preprocessor

Smart image preprocessor for e-ink picture frames. Uses local ML to detect art subjects in museum, gallery, and street art photos, then crops and zooms to highlight them for portrait display.

![Before and after samples](samples/hero_samples.jpg)

## Features

- **ML-powered smart cropping** -- YOLO-World + Grounding DINO ensemble detects art, sculptures, murals, and more
- **VLM fallback** -- Qwen3-VL-2B grounding pass via llama.cpp (~20s/image, cached) activates when YOLO/DINO are uncertain. Enabled by default
- **Text detection** -- EasyOCR filters signs, labels, and text-heavy regions from primary selection and secondary crops
- **Focal point detection** -- for large murals that fill the frame, a second Grounding DINO pass finds faces/figures inside the primary to use as the crop anchor
- **Contextual zoom** -- zooms in on small or distant subjects, leaves large ones untouched
- **Multi-crop** -- detects multiple art pieces and produces separate crops for each (enabled by default)
- **Batch processing** -- parallel workers with model caching
- **Local processing** -- no cloud dependencies, optional OpenVINO acceleration on Intel

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -e .
python scripts/download_models.py

# Process a single image (VLM + multi-crop enabled by default)
frame-prep process -i photo.jpg -o output/ -v

# Without VLM (faster, slightly lower accuracy)
frame-prep process -i photo.jpg -o output/ --no-vlm -v

# Batch process a directory
frame-prep batch -i ~/photos/art/ -o ~/photos/processed/ --skip-existing

# Only process files added/modified since a date (or within a range)
frame-prep batch -i ~/photos/art/ -o ~/photos/processed/ --since 2026-07-01
frame-prep batch -i ~/photos/art/ -o ~/photos/processed/ --since 2026-07-01 --until 2026-07-15
```

Output is 480x800 JPEG by default (3:5 portrait ratio for e-ink frames).

## More Samples

**Gallery art** -- painting and sculpture detection with smart crop:

![Gallery samples](samples/sample_gallery.jpg)

**Street art** -- rotated photo with subject detection:

![Street art sample](samples/sample_street_art.jpg)

**Focal point detection** -- wide mural fills the frame, second pass finds the face/figure to use as crop anchor:

![Focal detection sample](samples/sample_focal_detection.jpg)

**Text detection** -- EasyOCR filters text-heavy detections (signs, labels), selecting the actual artwork instead:

![Text detection sample](samples/sample_text_detection.jpg)

## Documentation

- **[Usage Reference](docs/USAGE.md)** -- full CLI options, cropping strategies, performance tuning
- **[Testing Guide](docs/TESTING_GUIDE.md)** -- quality assessment with interactive HTML reports
- **[Contextual Zoom](docs/CONTEXTUAL_ZOOM.md)** -- how zoom logic works
- **[Hardware Acceleration](docs/HARDWARE_ACCELERATION.md)** -- OpenVINO and threading optimization

## Quality Assessment

```bash
# Generate interactive detection report
frame-prep report

# Opens reports/interactive_detection_report.html
# Rate results, export feedback as JSON
```

![Interactive detection report](samples/report_screenshot.png)

Current accuracy: **94% IoU hit rate** (115/122) on ground truth test set (with `--vlm`; 88% without).

## Related Projects

- [onedrive-album-download](https://github.com/antoinecellerier/onedrive-album-download) -- download photo albums from OneDrive
- [librespot-epd-nowplaying](https://github.com/antoinecellerier/librespot-epd-nowplaying) -- Spotify now-playing display for e-ink frames

## License

[MIT](LICENSE)
