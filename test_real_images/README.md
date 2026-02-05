# Test Real Images

This directory contains test images and evaluation data for the picture frame preprocessor.

## Directory Structure

### Tracked in Git (Important Data)

- **input/** - Original test images (64 museum/gallery photos)
- **ground_truth_annotations.json** - Manual annotations with bounding boxes for 63 images
- **FEEDBACK_ANALYSIS.md** - Analysis of detection quality and improvements
- **TEST_SUMMARY.md** - Summary of test results

### Ignored by Git (Generated/Intermediate Results)

- **results/** - All intermediate test results (can be safely deleted and regenerated)
  - `intermediate_outputs/` - Various test runs with different models
  - `html_reports/` - Large HTML quality assessment reports

- **output_optimized/** - Final processed images using optimized ensemble (now the default)

## Regenerating Results

All generated outputs can be recreated:

```bash
# Regenerate output (uses optimized ensemble by default)
python scripts/batch_process.py \
  --input-dir test_real_images/input \
  --output-dir test_real_images/output_optimized

# Regenerate interactive detection report
python scripts/generate_interactive_report.py
```

## Safe to Delete

The `results/` and `output_*/` directories can be safely deleted to save space. They contain:
- Intermediate test outputs from model experiments
- Large HTML visualization reports
- Cached detection results

All of this data can be regenerated from the source images using the scripts.
