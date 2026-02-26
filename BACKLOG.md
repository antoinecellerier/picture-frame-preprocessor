# Backlog

## Session Summary (2026-02-26)

**Accuracy: 103/122 (84%) IoU hit rate** (after EXIF rotation fix in eval script).
Previous measured baseline was 94/122 (77%) — gap was purely due to eval script feeding rotated images.

### Completed this session
- Per-class accuracy evaluator: `scripts/evaluate_art_class_accuracy.py` (mural/mosaic/street_art/sculpture/painting/installation/non_art)
- Art class ground truth fully annotated: 122/122 images in `test_real_images/art_class_ground_truth.json`
- SigLIP2 class verifier implemented & evaluated — net -1, kept as `--siglip-verify` opt-in
- OWLv2 mosaic detection evaluated — ruled out (best F1=0.293 vs baseline 0.529)
- Mosaic failure root cause analysis (Space Invader pixel art vs trained-on-traditional-mosaics)
- EXIF rotation fix in `scripts/evaluate_art_class_accuracy.py`: **+9 IoU hits** (94→103)

### Attempted, net 0 or negative
- Size floor for mosaic/tile scoring: net -1 in both broad and targeted variants
- "pixel art" DINO prompt: net -4 (fires on murals, patterns, anything tile-like)
- Size bonus table tightening: net -1

### Remaining scoring failures (hard to fix without regressions)
- DSC_4371: mosaic (IoU=0.69, 5.0x) loses to art_installation (2.0x, large) — scoring gap
- DSC_4385: correct mosaic (IoU=0.84) loses to larger "figure" bbox — scoring gap

---

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

### 2. ~~Boost "street art" class multiplier~~ — DONE (2026-02-21)

Bumped to 3.5x (between specific-art 5.0x and scene-art 2.0x). Removed from `_scene_art_classes`, added explicit substring check in `_get_class_multiplier`.

### 3. Mosaic detection — root cause analysis (2026-02-26)

**The mosaics that fail are almost exclusively "Space Invader" pixel art** — small tile/pixel artworks
affixed to Parisian building walls by the artist Invader (and imitators). They look like video
game sprites made of colored tiles, NOT like traditional stone/ceramic mosaics.

**Updated failure breakdown after EXIF fix (10 IoU misses, 40% miss rate):**

Category A — Too small / not detected at all (3 images):
- DSC_3412 (smiley+sunflowers, ~1% area on stone building) — still miss
- DSC_4042 (ghost on white building, ~0.9% area) — still miss
- DSC_4388 (tiny figure at building corner, < 0.5% area) — still miss (borderline, iou≈0.15)

Category B — Detected but wrong location (6 images):
- DSC_4311, DSC_4312 (rocket, busy street scene)
- DSC_4020 (Sonic pixel art + stencil, multiple art pieces)
- 20210911_152658 (Minion, small, next to street signs)
- DSC_4385 (CORRECT mosaic at IoU=0.84 but loses to larger "figure" — SCORING)
- DSC_3401 (sculpture at IoU=0.87 but loses to competing bbox — SCORING)

Category C — Scoring problems only (2 images):
- DSC_4371: mosaic detected (IoU=0.69) but "art installation" wins
- DSC_4385: see Category B above

**FIXED by EXIF rotation (2026-02-26): DSC_4163 (Santa, was rotated 90°), DSC_4115 (crab alien),
plus 2 more mosaics and others across mural/sculpture/street_art classes.**

**Why passing mosaics work (15/25 after EXIF fix):**
- Figural mosaics (DSC_4303-4305): depicted figure large enough to detect as "sculpture/figure"
- Isolated colorful square on plain white wall (DSC_4302): high contrast → YOLO fires "mosaic"
- Key: contrast + isolation > size. DSC_4302 passes at 0.3% area on a plain wall;
  DSC_4388 fails at similar size in a cluttered urban scene.

**Root causes:**
1. Visual mismatch: YOLO/DINO "mosaic" class was trained on traditional stone/ceramic mosaics,
   not pixel art. The Space Invader pieces look like video game sprites, not ancient tile work.
2. Small size (1–5% of image) in complex urban scenes with competing architecture/signs
3. Image rotation: **FIXED** — eval script now applies exif_transpose; actual pipeline already did
4. Wrong-location detections: detector finds "mosaic" in the wrong tile/texture

**What's been tried:**
- OpenCV Hough Lines: FAILED (2026-02-25). Fires on text rows, shelves, facades.
- CLIP full-image: FAILED (2026-02-25). F1=0.403. Museum context swamps signal.
- CLIP region scoring: best result F1=0.529 (conf=0.10 candidates). Not good enough.
- OWLv2: FAILED (2026-02-26). Best F1=0.293. Fires on colorful murals equally.
- DINO prompt engineering: high-risk, multiple synonyms caused regressions (2026-02-22)
- Size bonus table tightening: net -1 (2026-02-26). Fixes 1-2, breaks 1.
- "pixel art" DINO prompt: FAILED (2026-02-26). Net -4. Fires on murals, patterns, anything tile-like.
- Size floor for mosaic/tile classes (targeted, 0.60): net -1 (2026-02-26). Fixes DSC_4371 but causes 2 new regressions where rogue small mosaic detections override correct primaries.
- **EXIF rotation fix in eval script: +9 (2026-02-26)!** The eval was testing rotated images. Pipeline already applied exif_transpose correctly. DSC_4163 and others now correctly detected.

**Prompt engineering lessons:**
- YOLO-World: zero-sum 28-class list, new prompts steal from existing
- DINO: safer but still risky; multiple mosaic synonyms caused regressions; "pixel art" also too broad

**Next candidates:**
1. **Scoring fix for DSC_4371/DSC_4385** — still pure scoring failures. Size floor variants all net ≤ 0 so far.
2. **Accept the hard cases** — DSC_4162 (0.1% area pig face), DSC_4042 (<1% area in cluttered scenes) are likely below any current detector's reliable threshold.

### 3a. Street sign mislabeling — HIGH VALUE false positive reduction

**Done (2026-02-26):** Added `'street name sign'`, `'road sign'`, `'name plate'`, `'street sign'`,
`'exit sign'`, `'speed limit'`, `'stop sign'` to `_avoid_classes` (0.05x). No effect on test set
because models never fire these class names on the actual signs.

**Root problem:** City street signs (Paris blue plaques, road signs, etc.) are mislabeled by both
YOLO-World and Grounding DINO as `'painting'`, `'artwork'`, `'decorative art'` — flat rectangular
coloured patches on walls look like art to the models. The avoid-list only fires when the model
emits a sign class name, which it currently doesn't.

**Affected images (from report):** 20200525, 20210911, DSC_3614, DSC_4042, DSC_4168, DSC_4305,
DSC_4382, DSC_4399 — all involve Parisian street name signs or traffic signs misclassified as art.

**Potential fixes (high value, worth pursuing):**
- **VLM region query**: ask Florence-2 / Moondream2 "is this a street sign or a piece of artwork?"
  on bbox crops where the primary is `painting` / `artwork` at low confidence (< 0.35). Only query
  ambiguous cases to limit latency.
- **Shape/colour heuristic**: Paris street signs are consistently narrow-aspect-ratio, blue/green
  rectangles with high text density. A lightweight classifier on the bbox crop could flag them.
- **CLIP disambiguation**: add "street sign with text", "informational sign", "directional sign"
  as negative prompts in a targeted CLIP check on painting/artwork detections below a conf threshold.

### 4. Person-as-art filtering (2 bad_crop)

When "painted figure" detection overlaps significantly with a person-shaped bbox and is small/off-center, deprioritize it. DSC_0001_BURST and 20210815_163856 both have actual people misclassified as "painted figure".

**SigLIPClassVerifier tried (2026-02-25) — FAILED:**
- Result: 88/116 (75.9%) vs 89/116 (76.7%) baseline — net -1
- Root problem: SigLIP2 zero-shot on a tight bbox crop **cannot distinguish "human depicted in artwork" from "real human"**. A mosaic figure, mural person, or clay figurine all score high on "person" prompts because SigLIP sees the visual form, not semantic context (painting vs real person).
- Protection list (`_CLEARLY_ART_KEYWORDS`) helps when detection class name is explicit (e.g., "mosaic figure") but fails for generic classes like "painted figure", "figure", "figure figne".
- `--siglip-verify` flag kept as an opt-in experiment; it's disabled by default and hurts accuracy when enabled.

**What would actually work:**
- Context-aware VLM query with the **full image + bounding box** — ask "is the person in [region] a real visitor or depicted in artwork?". Florence-2 or PaliGemma would be candidates for this.
- Or: confidence + position heuristics — real people tend to be at image edges, have lower detection confidence (< 0.30), and appear in multi-person groups.

### 5. Misclassification cleanup (3 bad_detection)

Hard problems requiring model-level improvements:
- Glass reflections (DSC_3065)
- Sign vs art (DSC_4042, DSC_4381)
- Partial mural fragments (DSC_4059)

---

## Alternative Models Evaluated (2026-02-25)

### Detection backbone alternatives to YOLO-World + Grounding DINO 1.5

| Model | Detection | Region Description | HF Available | Notes |
|-------|-----------|-------------------|--------------|-------|
| **DINO-X** (IDEA Research, Nov 2024) | ★★★★★ | ★★★★★ native | API-only | +5.8 AP on LVIS rare classes; unified region QA; not open-source yet |
| **OWLv2** (Google, 2024) | ★★★★★ | ✗ | ✓ `google/owlv2-large-patch14` | Better on small objects vs Grounding DINO; could help small mosaic detection |
| **Florence-2** (Microsoft, 2024) | ✗ (secondary) | ★★★★★ | ✓ `microsoft/Florence-2-large` (0.77B) | `<REGION_TO_DESCRIPTION>` takes full image + bbox → ideal for "painted vs real" |
| **Moondream2** | ✗ (secondary) | ★★★★★ VQA | ✓ `vikhyatk/moondream2` (2B/0.5B) | Ask "Is this a painted figure or real person?" on bbox region; very small |
| **YOLO11** (Ultralytics, 2024) | ★★★ | ✗ | ✓ | NOT open-vocabulary; YOLO-World still the standard |
| **YOLO26** (Ultralytics, Jan 2026) | ★★★★ | ✗ | Emerging | New open-vocabulary YOLO; evaluate when stable |

### Most promising next experiments

**For "painted figure vs real person" (items 4 + bad_crop):**
- **Florence-2 `<REGION_TO_DESCRIPTION>`**: Takes full image + bbox coordinates, returns region description preserving context. Parse output for "painting", "sculpture", "artwork" vs "visitor", "person". Only need to query the primary detection.
- **Moondream2 VQA (0.5B)**: Ask "Is the person in this image a real person or depicted in artwork?" on the bbox crop. Fast at 0.5B, fine-tuned for fine-grained distinctions.

**For small mosaic detection (items 3 + bad_detection):**
- **OWLv2**: EVALUATED (2026-02-26) — does not beat baseline. Best F1=0.293 (art_only prompts) vs baseline F1=0.529. Root cause: assigns high scores to colorful murals/frescoes — no clean threshold exists. Not worth integrating.
- **Next direction**: Expand scope to all art classes using the new `art_class_ground_truth.json`. Build a multi-class evaluator and test models on the full taxonomy (painting/mural/mosaic/sculpture/ceramic/street_art/installation). Accept fuzzy boundaries (mural vs street_art vs painting is sometimes ambiguous).

**For full detection upgrade:**
- **DINO-X**: Wait for open-source release. Currently API-only via IDEA Research SDK. Would be a direct drop-in upgrade to Grounding DINO 1.5 with native region QA.

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
