# Contextual Zoom Logic

## Overview

The zoom feature is **contextual** - it only zooms when needed based on the detected subject size, not as a fixed operation.

## How It Works

### 1. Subject Size Detection

When YOLO detects objects:
1. Calculate subject area (from bounding box)
2. Calculate crop area
3. Compute ratio: `subject_area / crop_area`

### 2. Contextual Zoom Calculation

**Target**: Make subject fill ~70% of the frame's larger dimension for optimal viewing

| Subject Ratio | Subject Size | Zoom Applied | Logic |
|---------------|--------------|--------------|-------|
| > 65% | Large | **No zoom** | Subject already fills frame |
| 45-65% | Medium | **Up to 1.2x** | Slight zoom to remove excess |
| 25-45% | Small | **Moderate** | Zoom to reach ~70% fill (capped at `--zoom`) |
| < 25% | Tiny | **Aggressive** | Max zoom (capped at `--zoom`) |

### 3. Examples

**Example 1: Large Subject (No Zoom)**
```
Input: 4000x3000 image
Subject: 2000x2000 sculpture (area = 4M pixels)
Crop: 1800x3000 (area = 5.4M pixels)
Ratio: 4M / 5.4M = 0.74 (74%)
Result: No zoom (subject already large)
```

**Example 2: Small Subject (Moderate Zoom)**
```
Input: 4000x3000 image
Subject: 600x600 painting (area = 360K pixels)
Crop: 1800x3000 (area = 5.4M pixels)
Ratio: 360K / 5.4M = 0.067 (6.7%)
Needed: sqrt(0.65 / 0.067) = 3.1x
Result: Apply 1.3x zoom (capped at max_zoom)
```

**Example 3: Medium Subject (Slight Zoom)**
```
Input: 4000x3000 image
Subject: 1200x1400 mural (area = 1.68M pixels)
Crop: 1800x3000 (area = 5.4M pixels)
Ratio: 1.68M / 5.4M = 0.31 (31%)
Needed: sqrt(0.65 / 0.31) = 1.45x
Result: Apply 1.3x zoom (within limits)
```

## Saliency Fallback

When YOLO finds nothing, saliency detection is used:
- **No exact subject size** available
- Apply **conservative zoom** (max 1.2x)
- Safer approach since we don't know boundaries

## Configuration

### Default Settings
```bash
--zoom 8.0  # Maximum zoom cap (default, very aggressive for tiny subjects)
```

The actual zoom applied depends on subject size -- large subjects get no zoom, tiny subjects can zoom up to the cap.

### Custom Maximum Zoom
```bash
# Conservative zoom (cap at 1.3x)
--zoom 1.3

# Disable zoom completely
--zoom 1.0

# Moderate zoom
--zoom 2.0
```

## Algorithm Pseudocode

```python
def calculate_contextual_zoom(max_dim_ratio, max_zoom):
    """
    max_dim_ratio: max(width_ratio, height_ratio) of subject vs crop
    max_zoom: user-configured maximum (e.g., 8.0)
    """
    target = 0.70  # Subject should fill ~70% of frame

    if max_dim_ratio >= 0.65:
        return 1.0  # No zoom needed

    elif max_dim_ratio >= 0.45:
        zoom_needed = target / max_dim_ratio
        return min(zoom_needed, 1.2, max_zoom)  # Cap at 1.2x

    elif max_dim_ratio >= 0.25:
        zoom_needed = target / max_dim_ratio
        return min(zoom_needed, max_zoom)

    else:
        # Aggressive zoom for tiny subjects
        zoom_needed = target / max(max_dim_ratio, 0.05)
        return min(zoom_needed, max_zoom)
```

## Benefits

1. **Smart**: Only zooms when actually needed
2. **Proportional**: Zoom amount based on subject size
3. **Safe**: Capped at user maximum, won't over-zoom
4. **Addresses feedback**:
   - ✅ "Zooming in could help" → Auto-detects small subjects
   - ✅ "Too much wall/background" → Removes excess when subject is small
   - ✅ "Focus on key features" → Zooms on detected art

## Edge Cases

### No Detections
- Falls back to saliency (moderate 1.2x zoom)
- If saliency fails, center crop (no zoom)

### Multiple Detections
- Selects primary subject via center-weighting and class priorities (not just highest confidence)
- Centers on primary, zoom based on its size

### Very Large Subjects
- Ratio > 0.6 → No zoom applied
- Preserves full subject visibility

### Very Small Subjects
- Ratio < 0.05 → Caps at max_zoom
- Prevents excessive pixelation

## Testing

To verify contextual zoom is working:

### Using the Interactive Detection Report

```bash
frame-prep report
```

The report (`reports/interactive_detection_report.html`) provides:

1. **Configuration Summary**: Shows max zoom factor and other parameters
2. **Side-by-side Comparison**: Each image displays:
   - Detection view: Original with bounding boxes
   - Result view: Cropped output with zoom factor label (e.g., "1.30x")
3. **Metadata**: Zoom factor shown in each card's header

### What to Look For

| Detected Subject | Expected Zoom | Visual Check |
|------------------|---------------|--------------|
| Large (fills frame) | 1.00x | No zoom applied |
| Medium | 1.00x - 1.20x | Slight tightening |
| Small | 1.20x+ | Noticeable zoom |
| Tiny | Up to `--zoom` cap | Aggressive zoom |

### Verifying Behavior

1. Open the report in browser
2. Filter by detection count to find images with varying subject sizes
3. Compare zoom values:
   - Large subjects → `1.00x` (no zoom)
   - Small subjects → `1.30x` (max zoom)
4. Verify cropped result appropriately frames the subject
