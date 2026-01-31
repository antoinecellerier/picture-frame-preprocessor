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

**Target**: Make subject fill ~60-70% of the frame for optimal viewing

| Subject Ratio | Subject Size | Zoom Applied | Logic |
|---------------|--------------|--------------|-------|
| > 60% | Large | **No zoom** | Subject already fills frame |
| 40-60% | Medium | **1.15x** | Slight zoom to remove excess |
| 20-40% | Small | **Moderate** | Zoom to reach ~65% fill |
| < 20% | Tiny | **Aggressive** | Max zoom (up to limit) |

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
--zoom 1.3  # Maximum zoom factor
```

The actual zoom applied will be **≤ 1.3x** based on subject size.

### Custom Maximum Zoom
```bash
# Allow up to 1.5x zoom for tiny subjects
--zoom 1.5

# Disable zoom completely
--zoom 1.0

# Conservative zoom
--zoom 1.2
```

## Algorithm Pseudocode

```python
def calculate_contextual_zoom(subject_ratio, max_zoom):
    """
    subject_ratio: 0.0 to 1.0 (subject area / crop area)
    max_zoom: user-configured maximum (e.g., 1.3)
    """

    if subject_ratio >= 0.6:
        return 1.0  # No zoom needed

    elif subject_ratio >= 0.4:
        return min(1.15, max_zoom)  # Slight zoom

    elif subject_ratio >= 0.2:
        # Moderate zoom to reach target
        zoom_needed = sqrt(0.65 / subject_ratio)
        return min(zoom_needed, max_zoom)

    else:
        # Aggressive zoom for tiny subjects
        zoom_needed = sqrt(0.65 / max(subject_ratio, 0.01))
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
- Uses primary (highest confidence) detection
- Centers on primary, zoom based on its size

### Very Large Subjects
- Ratio > 0.6 → No zoom applied
- Preserves full subject visibility

### Very Small Subjects
- Ratio < 0.05 → Caps at max_zoom
- Prevents excessive pixelation

## Testing

To verify contextual zoom is working:

1. Check ML visualization HTML
2. Look for subject size info
3. Compare zoom applied:
   - Large subjects → minimal/no zoom
   - Small subjects → noticeable zoom
   - Matches the ratios above
