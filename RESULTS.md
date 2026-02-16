# Picture Frame Preprocessor - Results Summary

## Final Model Performance

Evaluated on 63 museum/gallery images with 142 ground truth bounding boxes.

| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| YOLOv8m (baseline) | 38.1% | ~0.1s/img | 24/63 images, 39 failures |
| YOLO-World | 52.4% | ~0.7s/img | 33/63 images, 30 failures |
| Ensemble (YOLO+DETR) | 63.5% | ~3.5s/img | 40/63 images, 23 failures |
| Grounding DINO | 88.9% | ~12s/img | 56/63 images, 7 failures |
| **🏆 Optimized Ensemble** | **98.4%** | **~13s/img** | **62/63 images, 1 failure** |

## Optimized Ensemble Configuration

**Best performing configuration** for art/museum image detection:

- **Models**: YOLO-World (improved prompts) + Grounding DINO
- **Merge threshold**: 0.2 (optimized for maximum recall)
- **Confidence threshold**: 0.25
- **YOLO-World prompts**: 23 contextual art-specific classes
  - Examples: "museum exhibit", "gallery display", "sculpture on pedestal", "framed artwork"
- **Grounding DINO prompts**: 12 art-focused categories
  - Examples: "sculpture", "statue", "painting", "art installation", "mosaic"

## Key Findings

### 1. Ensemble Superiority
Combining YOLO-World and Grounding DINO achieves significantly better results than either model alone:
- YOLO-World catches fast-moving objects and general art pieces
- Grounding DINO provides high-accuracy detection for complex scenes
- Low merge threshold (0.2) ensures maximum detection coverage

### 2. Prompt Engineering Impact
Improved contextual prompts for YOLO-World increased accuracy:
- Original prompts: 52.4% accuracy
- Improved contextual prompts: 54.0% accuracy (+1.6%)
- Combined with Grounding DINO: 98.4% accuracy

### 3. Only One Failure
The optimized ensemble failed on only 1 image out of 63:
- **20191124_100445.jpg**: Timeout error (processing took too long)
- All other 62 images successfully detected

## Production Recommendations

### For Maximum Accuracy (Default)
```bash
frame-prep process --input image.jpg --output out/
```
- Uses optimized ensemble by default (98.4% accuracy)
- Suitable for offline batch processing
- ~13 seconds per image

### For Faster Processing
```bash
frame-prep process --input image.jpg --output out/ --single-model
```
- Use single YOLOv8m for faster processing
- ~0.7 seconds per image
- Good for interactive applications

### For Balanced Performance
```bash
frame-prep process --input image.jpg --output out/ --ensemble
```
- Use ensemble detector (63.5% accuracy)
- ~3.5 seconds per image
- Good middle ground

## Hardware Optimization

Tested on: Intel Core i7 (16 threads), 32GB RAM, Intel Iris Xe GPU

**Optimizations applied:**
- OpenVINO acceleration for YOLO models
- Multi-worker batch processing (4 workers recommended)
- Caching of model weights
- Intel PyTorch Extension (IPEX) support (disabled due to version incompatibility)

**Batch processing speedup:**
- Single worker: ~13s per image
- 4 workers: ~3.25s per image (4x speedup)

## Viewing Results

Generated reports are available in `reports/` directory:
- `final_model_comparison.html` - Visual comparison of all models with bounding boxes
- Other HTML reports - Various quality and performance assessments

To regenerate reports:
```bash
frame-prep report
```

## Next Steps

See `FINAL_RECOMMENDATIONS.md` for:
- Further optimization opportunities
- Additional model experiments
- Production deployment guidelines
