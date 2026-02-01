# Testing Guide

This guide explains how to use the test utilities to evaluate crop quality on sample images.

## Test Utilities

### 1. Generate Test Set (`scripts/generate_test_set.py`)

Creates a random sample of images from a source directory for testing.

**Usage:**
```bash
python scripts/generate_test_set.py \
  --source ~/stuff/onedrive-album-download/downloads/art/ \
  --output test_real_images/input \
  --count 64 \
  --seed 42
```

**Options:**
- `--source, -s`: Source directory containing images (required)
- `--output, -o`: Output directory for test set (required)
- `--count, -n`: Number of images to select (default: 64)
- `--seed`: Random seed for reproducibility (optional)

**Use Cases:**
- Create representative test sets for quality assessment
- Reproducible test sets with seed parameter
- Quick sampling from large image collections

### 2. Generate Interactive Detection Report (`scripts/generate_interactive_report.py`)

Creates a comprehensive HTML report showing detection results alongside cropped outputs for quality assessment.

**Usage:**
```bash
python scripts/generate_interactive_report.py
```

**Features:**
- **Configuration Summary**: Shows all detection and cropping parameters at the top
  - Detection Strategy: Models used (YOLO-World + Grounding DINO)
  - Detection Parameters: Confidence threshold, merge threshold
  - Cropping Strategy: Target dimensions, zoom factor, fallback settings
- **Side-by-side Comparison**: Each image shows:
  - Detection view (left): Original image with bounding boxes
  - Result view (right): Actual cropped output with zoom factor
- **Ground Truth Comparison**: Compares detections against annotated ground truth
- **Accuracy Metrics**: IoU scores and overall accuracy percentage
- **Feedback System**: Rate results as Good/Poor/Zoom Issue with comments
- **Export**: Save feedback as JSON for analysis

**Output:** `reports/interactive_detection_report.html`

**Configuration Parameters Displayed:**
| Section | Parameter | Description |
|---------|-----------|-------------|
| Detection | Ensemble | OptimizedEnsembleDetector |
| Detection | Models | YOLO-World (yolov8m-worldv2) + Grounding DINO (tiny) |
| Detection | Confidence Threshold | Minimum detection confidence (default: 0.25) |
| Detection | Merge Threshold | IoU threshold for merging boxes (default: 0.2) |
| Cropping | Target Dimensions | Output size (default: 1080×1920, 9:16 aspect) |
| Cropping | Max Zoom Factor | Maximum zoom applied (default: 1.3x) |
| Cropping | Saliency Fallback | Use saliency when no detections (default: enabled) |

### 3. Generate Quality Report (`scripts/generate_quality_report.py`)

Creates an interactive HTML report for assessing crop quality with rating system.

**Usage:**
```bash
python scripts/generate_quality_report.py \
  --input-dir test_real_images/input/ \
  --output-dir test_real_images/output/ \
  --html test_real_images/quality_assessment.html \
  --title "Crop Quality Assessment - 64 Images"
```

**Options:**
- `--input-dir`: Directory with original images (required)
- `--output-dir`: Directory with processed images (required)
- `--html`: Path to save HTML report (default: quality_report.html)
- `--title`: Report title (default: Crop Quality Assessment)

**Features:**
- ⭐ Rate each crop from 1-5 stars
- 💬 Add optional comments per image
- 📊 Live statistics (total, rated, average rating)
- 💾 Export feedback as JSON
- 🔄 Auto-saves to browser localStorage (survives page refresh)
- 📱 Responsive design for desktop/tablet viewing

## Complete Workflow

### Step 1: Generate Test Set

```bash
# Create 64-image random sample
python scripts/generate_test_set.py \
  --source ~/stuff/onedrive-album-download/downloads/art/ \
  --output test_real_images/input \
  --count 64 \
  --seed 42
```

### Step 2: Process Images

```bash
# Process with smart strategy (default)
python scripts/batch_process.py \
  --input-dir test_real_images/input/ \
  --output-dir test_real_images/output/ \
  --width 480 --height 800 \
  --workers 4
```

### Step 3: Generate Quality Report

```bash
# Create interactive HTML assessment
python scripts/generate_quality_report.py \
  --input-dir test_real_images/input/ \
  --output-dir test_real_images/output/ \
  --html test_real_images/quality_assessment.html
```

### Step 4: Review and Rate

1. Open `quality_assessment.html` in your browser
2. Review each image comparison
3. Rate crop quality (1-5 stars)
4. Add comments for problematic crops
5. Export feedback as JSON

### Step 5: Analyze Results

The exported JSON contains:
```json
{
  "timestamp": "2026-01-30T...",
  "totalImages": 64,
  "ratedImages": 64,
  "feedback": {
    "1": {
      "rating": 5,
      "filename": "image1.jpg",
      "comment": "Perfect crop"
    },
    "2": {
      "rating": 3,
      "filename": "image2.jpg",
      "comment": "Subject slightly cut off"
    }
  }
}
```

## Quality Rating Guidelines

### ⭐⭐⭐⭐⭐ Excellent (5 stars)
- All important subjects preserved
- Good composition
- Nothing important cut off
- Visually appealing crop

### ⭐⭐⭐⭐ Good (4 stars)
- Main subjects preserved
- Minor elements may be cropped
- Overall satisfactory result

### ⭐⭐⭐ Fair (3 stars)
- Main subject partially preserved
- Some important elements cropped
- Acceptable but not ideal

### ⭐⭐ Poor (2 stars)
- Important subjects cut off
- Poor composition
- Significant quality loss

### ⭐ Bad (1 star)
- Critical subjects missing
- Unusable crop
- Wrong area selected

## Analyzing Feedback

### Calculate Statistics

```python
import json

with open('crop_quality_feedback_2026-01-30.json') as f:
    data = json.load(f)

ratings = [f['rating'] for f in data['feedback'].values() if 'rating' in f]

print(f"Total rated: {len(ratings)}")
print(f"Average: {sum(ratings)/len(ratings):.2f}")
print(f"Excellent (5): {ratings.count(5)}")
print(f"Good (4): {ratings.count(4)}")
print(f"Fair (3): {ratings.count(3)}")
print(f"Poor (2): {ratings.count(2)}")
print(f"Bad (1): {ratings.count(1)}")
```

### Find Problematic Crops

```python
# Images rated 2 or below
problematic = [
    (id, f['filename'], f.get('comment', ''))
    for id, f in data['feedback'].items()
    if f.get('rating', 0) <= 2
]

for img_id, filename, comment in problematic:
    print(f"{filename}: {comment}")
```

## Testing Different Strategies

Compare different cropping strategies:

```bash
# Smart strategy (default)
python scripts/batch_process.py \
  -i test_real_images/input/ \
  -o test_real_images/smart/ \
  --strategy smart

# Saliency strategy
python scripts/batch_process.py \
  -i test_real_images/input/ \
  -o test_real_images/saliency/ \
  --strategy saliency

# Center strategy
python scripts/batch_process.py \
  -i test_real_images/input/ \
  -o test_real_images/center/ \
  --strategy center

# Generate comparison reports
python scripts/generate_quality_report.py \
  --input-dir test_real_images/input/ \
  --output-dir test_real_images/smart/ \
  --html smart_quality.html \
  --title "Smart Strategy"

python scripts/generate_quality_report.py \
  --input-dir test_real_images/input/ \
  --output-dir test_real_images/saliency/ \
  --html saliency_quality.html \
  --title "Saliency Strategy"
```

## Reproducible Testing

Use the same seed for reproducible test sets:

```bash
# Everyone gets the same 64 images
python scripts/generate_test_set.py \
  --source ~/images/ \
  --output test_set/ \
  --count 64 \
  --seed 42
```

## Tips

1. **Start small**: Test with 8-16 images first
2. **Increase gradually**: Move to 64 for comprehensive testing
3. **Use seeds**: Reproducible results for comparisons
4. **Export regularly**: Save feedback JSON periodically
5. **Note patterns**: Look for systematic issues in comments
6. **Test edge cases**: Include various image types (portraits, landscapes, abstract)

## Automation

Create a testing script:

```bash
#!/bin/bash
# test_pipeline.sh

# Generate test set
python scripts/generate_test_set.py \
  -s ~/art/ -o test/input -n 64 --seed 42

# Process images
python scripts/batch_process.py \
  -i test/input -o test/output

# Generate report
python scripts/generate_quality_report.py \
  --input-dir test/input \
  --output-dir test/output \
  --html quality_report.html

# Open in browser
xdg-open quality_report.html
```

Make it executable:
```bash
chmod +x test_pipeline.sh
./test_pipeline.sh
```
