# Crop Quality Feedback Analysis

**Date**: 2026-01-30
**Images Reviewed**: 64
**Average Rating**: 3.53/5.0

## Summary

✅ **Strengths:**
- 57.8% of crops rated Good-Excellent (4-5 stars)
- 17 images received perfect 5-star ratings
- EXIF orientation handling working correctly

❌ **Weaknesses:**
- 42.2% need improvement (1-3 stars)
- 11 problematic crops (1-2 stars)
- Several critical issues with subject detection

## Rating Distribution

| Stars | Count | Percentage |
|-------|-------|------------|
| ⭐⭐⭐⭐⭐ | 17 | 26.6% |
| ⭐⭐⭐⭐ | 20 | 31.2% |
| ⭐⭐⭐ | 16 | 25.0% |
| ⭐⭐ | 2 | 3.1% |
| ⭐ | 9 | 14.1% |

## Top Issues (from 64 comments)

1. **Needs more zoom** (12 mentions)
   - Crops are too wide, showing unnecessary wall/frame
   - Should zoom in tighter on art subjects
   - Example: "zooming in could work to improve the result"

2. **Off-center crops** (9 mentions)
   - ML detecting wrong subjects
   - Art is in image center but crop is off
   - Example: "The key painted cabinet feature is completely offcenter"

3. **Too wide to crop** (8 mentions)
   - Panoramic/wide art doesn't fit 3:5 aspect ratio
   - Need different strategy for ultra-wide subjects
   - Example: "the original art is too wide"

4. **Missing key subjects** (7 mentions)
   - ML not detecting the main art/sculpture
   - Crop completely misses the artwork
   - Example: "The crop completely misses the face painting in the center"

5. **Should focus on specific feature** (6 mentions)
   - When full art doesn't fit, zoom on key detail
   - Example: "focus on a smaller crop area on key features"

## Critical Failures (1-star ratings)

### Complete Miss Failures
1. **DSC_4382.JPG** - "completely misses the face painting in the center"
2. **DSC_1734.JPG** - "completely missing the sculpture despite it being in the center"
3. **DSC_2846.JPG** - "key tile art figure completely missing"
4. **20210815_163856.jpg** - "misses the small green bird figure in the center"
5. **DSC_4107.JPG** - "'la vache qui rit' art figure almost completely missing"

### Off-center Failures
6. **DSC_0312.JPG** - "key painted cabinet feature is completely offcenter"
7. **20201003_173051.jpg** - "misses the sculpture with two people"

### Strategy Failures
8. **20130317_020501_Android.jpg** - Image too wide, needs zoom on key features
9. **DSC_4063.JPG** - Not art (label closeup), should be filtered out

## Root Cause Analysis

### Why ML is Failing

**Hypothesis 1: Wrong subjects detected**
- YOLO trained on COCO (people, cars, animals, common objects)
- Art/sculptures/murals may not be detected as discrete objects
- ML might detect background people/objects instead of the art

**Hypothesis 2: Center-crop fallback issues**
- When no objects detected, falls back to center crop
- But some centered art still gets cropped incorrectly
- Suggests crop calculation or aspect ratio handling issues

**Hypothesis 3: No zoom/scale adjustment**
- Crop maintains full height of image
- Doesn't zoom in to exclude wall/frame/context
- Should potentially crop tighter on detected subjects

## Recommendations

### 1. **Add ML Visualization** ✅ Ready
Deploy the ML visualization HTML to see:
- What objects are being detected
- Which detection is used as primary
- Exact crop area boundaries
This will help diagnose why crops are off

### 2. **Improve Detection Strategy**
- Use art-specific detection (fine-tuned model?)
- Add saliency-based subject detection for art
- Combine ML + saliency for better subject finding

### 3. **Add Smart Zoom**
- When detected subject is small, zoom in 1.2-1.5x
- Remove excessive wall/frame/background
- Configurable zoom factor per image type

### 4. **Better Centering Logic**
- When multiple subjects detected, find center of mass
- For center-crop fallback, use saliency to find actual center of interest
- Add "assume center contains art" heuristic

### 5. **Handle Wide Images Differently**
- Detect ultra-wide aspect ratios (>3:1)
- Use different crop strategy: zoom on key feature
- Or generate multiple crops from one wide image

### 6. **Add Content Filtering**
- Detect and skip non-art images (text labels, etc.)
- Use image classification to identify art vs. non-art

## Next Steps

1. **Deploy ML visualization** to understand current behavior
2. **Identify patterns** in what ML is detecting
3. **Tune detection parameters** (confidence, model choice)
4. **Implement zoom feature** for tighter crops
5. **Test improved strategy** on problematic images
6. **Re-evaluate** on full 64-image set

## Specific Image Issues to Review with ML Viz

High priority for visualization analysis:
- DSC_4382.JPG - Missing face painting
- DSC_1734.JPG - Missing sculpture
- DSC_2846.JPG - Missing tile art
- DSC_0312.JPG - Offcenter cabinet
- 20210815_163856.jpg - Missing green bird

These will show us what (if anything) ML detected and why it chose that crop area.
