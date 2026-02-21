# Backlog

## Session Summary (2026-02-21)

**Current accuracy: TBD** (focal detection added, report regenerated). Feedback: 94 good, 25 bad (119 reviewed / 122 total — `detection_feedback_2026-02-21T19-06-31-369Z.json`).

### Completed this session
- Focal point detection pass: Grounding DINO on primary's zone with face/figure prompts
- Focal dets passed separately to cropper (never merged into main detections) to prevent class-multiplier corruption of primary selection
- Parabolic area scoring for inner anchors: `conf × 4r(1-r)`, peaks at 50% of primary — replaces hard 65% cap and sqrt scoring
- 3D art skip: focal pass and inner anchor logic both skip 3D primaries (sculpture, statue, etc.)
- Report: focal dets in magenta, selected anchor in gold; config shows focal detection setup; removed Ctrl+F shortcut
- Saved feedback: `test_real_images/detection_feedback_2026-02-21T19-06-31-369Z.json`

### New issues from feedback
- `DSC_1734.JPG`, `DSC_2149.JPG`: Background "framed artwork" used as anchor for statue primary — `is_3d_art` skip was only in preprocessor, not in `_get_quality_inner_detections` inside cropper
- `20210530_135908.jpg`: "mosaic" appearing as bad first crop in multi-crop display
- `20210808_162451.jpg`: Exhibit label detected as mosaic, appears as bad first crop

---

## Session Summary (2026-02-16)

### Completed this session
- Split feedback buttons: bad → bad_detection / bad_crop / bad_both
- Multi-crop: primary subject always first, wide primaries use inner focal points
- Crop target highlighting (orange) in detection image
- Secondary crop quality filters: class_mult >= 2.0, edge/size rejection
- Edge penalty in primary scoring: 0.6x for 1 edge, 0.4x for 2+ edges
- Saved feedback: `reports/feedback/detection_feedback_2026-02-15T22-56-09-832Z.json`

---

## Remaining Issues (30 items from latest feedback)

### Bad Detection — Wrong primary selection (8 cases)

The correct subject exists in detections but a different one wins primary scoring.

**Small central art beaten by higher-confidence non-art (3):**
- `DSC_4388.JPG`: "framed artwork" (0.45, small, off-center) beats "art street art" (0.31, central). Class multiplier mismatch: 5.0x vs 2.0x. Needs "street art" boost or context-aware scoring.
- `DSC_4042.JPG`: "painting" (0.32, actually a sign) beats "decorative art" (0.28, actual mosaic). Misclassification — sign detected as "painting".
- `DSC_0155.JPG`: "painted figure" (0.34) beats GT vase area. Wrong region entirely.

**Large bbox same-tier competition (3):**
- `DSC_4291.JPG`: "mosaic" (0.32) selected, GT wants "figure figne" (garbled class → default 1.5x)
- `DSC_4385.JPG`: "figure" (28% area, 0.26) beats "mosaic" (1.8%, 0.30). Large bbox with 1.2x size bonus
- `DSC_4381.JPG`: "decorated sign" (0.33) beats "sculpture" (0.25). Sign-art confusion.

**Partial / wrong detection (2):**
- `DSC_0274.JPG`: "painted figure" (0.28) picks wrong region of a tile mural
- `DSC_3367.JPG`: "painted figure" (0.28) only partially covers the subject

### Bad Detection — Subject not detected at all (6 cases)

Models fail to find the actual art subject. Would need model improvements, additional prompts, or post-processing.

- `20200525_170722.jpg`: Sculpture/statue figures (0.30-0.31 conf) detected but not selected — actually a primary selection issue, mosaic (0.39) wins
- `20210911_152658.jpg`: Minion mosaic not detected by any model
- `DSC_0493.JPG`: Large chalk drawing of woman's face → "painted figure" (0.44) picks tiny detail instead
- `DSC_4162.JPG`: Tiny pig face mosaic → only detection is "decorated sign" at edge
- `DSC_4201.JPG`: 3 cartoon figures in mural not individually detected
- `DSC_4311.JPG` / `DSC_4312.JPG`: Mosaic rocket tile art not properly detected

### Bad Detection — Misclassification (3 cases)

- `DSC_3065.JPG`: Reflection in glass detected as "painted figure"
- `DSC_4059.JPG`: "painted figure" picks wrong fragment of PARIS mural
- `20210910_204401.jpg`: FIXED by edge penalty (was bad_detection in earlier round)

### Bad Crop — Wide primary, suboptimal framing (6 cases)

Primary is correctly detected as a large mural/painting but the crop doesn't focus on the interesting part.

- `DSC_0153.JPG`: Wide mural — crop should focus on face/focal area
- `DSC_1488.JPG`: Wide mural — crop should frame the "face" area
- `20210530_135908.jpg`: Wide mural — lion's head (crop 3) is best but not first
- `DSC_4205.JPG`: Wide mural — should use inner figure detections as focal points
- `20210808_162451.jpg`: Huge mural is primary, should produce a good single crop
- `DSC_1045.JPG`: Overlapping figurine detections — wrong one centered

**Potential fix:** Use saliency analysis within the primary bbox to find the most visually interesting focal point when no good inner detections exist.

### Bad Crop — Wrong primary leads to bad crop (5 cases)

Root cause is in detection, not cropping. Fix would propagate from better primary selection.

- `DSC_0001_BURST20241121142123881.JPG`: "painted figure" (0.45) is a person walking in snow
- `DSC_4381.JPG`: "decorated sign" primary instead of mosaic/sculpture
- `DSC_4385.JPG`: "figure" primary instead of "mosaic"
- `DSC_4294.JPG`: "mosaic" (0.25) primary is tiny; exhibit/art installation should win
- `DSC_4371.JPG`: "art installation" primary, small mosaic is the actual subject

### Bad Crop — Junk secondary crop (3 cases)

Secondary crop target is not visually interesting art.

- `20210213_154948.jpg`: "painted figure" (0.31) is a building behind a fence
- `DSC_0312.JPG`: "vase" (0.45) is a plant box — misclassification by model
- `DSC_3401.JPG`: Multiple overlapping sculpture detections, confusing result

### Bad Crop — Other (2 cases)

- `DSC_4312.JPG`: Primary mosaic correct but GT expects a different area (rocket)
- `20210815_163856.jpg`: "art installation exhibit" primary, woman walking selected as secondary

---

## Investigation Roadmap

### 1. Saliency-guided focal point for wide primaries (6 bad_crop)

When primary bbox is wider than crop and no quality inner detections exist, use saliency map within the primary bbox to find the most visually interesting crop anchor. Would help DSC_0153, DSC_1488, 20210530_135908, DSC_4205, 20210808_162451, DSC_1045.

**Approach:** Run composition analyzer on the primary bbox region, pick highest-saliency point as crop anchor.

### 2. Boost "street art" class multiplier (2 bad_detection)

"street art" currently gets 2.0x (scene_art tier). Several images have central street art losing to less-relevant 5.0x detections. Consider bumping to 3.0-3.5x, or add context: if the image has many street-level detections, boost street art classes.

**Risk:** Could cause regressions where actual "street art" label is a misdetection (sign, lamp post).

### 3. YOLO-World prompt engineering for mosaics (4 bad_detection)

Several small mosaic/tile art pieces go undetected. Try adding specific prompts:
- "small mosaic tile", "decorative tile", "tile artwork on wall"
- "chalk drawing on ground", "pavement art"

### 4. Person-as-art filtering (2 bad_crop)

When "painted figure" detection overlaps significantly with a person-shaped bbox and is small/off-center, deprioritize it. DSC_0001_BURST and 20210815_163856 both have actual people misclassified as "painted figure".

### 5. Misclassification cleanup (3 bad_detection)

Hard problems requiring model-level improvements:
- Glass reflections (DSC_3065)
- Sign vs art (DSC_4042, DSC_4381)
- Partial mural fragments (DSC_4059)

---

## Previously Completed

### DONE: Art-class priority in primary selection (2026-02-07)
Three-tier class scoring. Accuracy: 62.1% → 72.4%.

### DONE: Fix zoom level and centering (2026-02-07)
`ZOOM_FACTOR` 1.3 → 8.0, fixed smart zoom centering.

### DONE: Non-art image filtering (2026-02-07)
Art score heuristic to skip non-art images.

### DONE: Multi-crop for panoramic scenes (2026-02-07)
`--multi-crop` flag, `crop_all_subjects` in cropper.

### DONE: Containment/nesting logic — REJECTED (2026-02-15)
Regressed 72.4% → 58.6%. Large art detections ARE the subject.

### DONE: Edge penalty in primary scoring (2026-02-16)
0.6x/0.4x penalty for edge-touching detections. Fixed 2 images, no regressions.

### DONE: Multi-crop ordering and secondary filtering (2026-02-16)
Primary first, quality filters for secondaries, crop target highlighting.
