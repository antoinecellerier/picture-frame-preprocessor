# Backlog

## Detection Quality

### DONE: Boost art-class priority in primary selection
Implemented 2026-02-07. Three-tier class scoring (specific art 5.0x > scene art 2.0x > generic scene 0.3x), revised size bonus, deprioritized "exhibit"/"display". Accuracy improved from 62.1% (72/116) to 72.4% (84/116). +17 fixes, -5 regressions.

### DONE: Fix zoom level and centering
Implemented 2026-02-07. Increased `ZOOM_FACTOR` from 1.3 to 8.0, fixed `_apply_smart_zoom` to center on subject position instead of crop center. All 5 zoom-flagged images now zoom correctly.

### Latest feedback: 88/119 good (74%), 22 bad, 9 other
Up from 70/119 (59%) before scoring+zoom fixes. Net +18 improvements (23 fixes, 4 regressions).

### P0: Wrong primary selection (10 of 22 bad)
Correct detection exists but wrong one is selected as primary.

**Large bbox same-tier competition (6 cases):**
When multiple detections are in the same scoring tier (both specific_art 5.0x), the larger/higher-confidence one wins despite a smaller, more precise detection matching ground truth.
- `DSC_3401.JPG`: "sculpture statue" (39% area, conf=0.32) beats "sculpture on pedestal" (1%, conf=0.46)
- `DSC_4385.JPG`: "figure" (28%, conf=0.26) beats "mosaic" (1.8%, conf=0.30)
- `DSC_4371.JPG`: "art installation" (large) beats precise "mosaic"
- `20201003_173051.jpg`: "sculpture on pedestal" (wrong position) beats correct area — REGRESSION
- `20220219_145944.jpg`: "mosaic" (21%, top of image) beats central painted figures — REGRESSION
- `20210911_152658.jpg`: "sculpture statue painted figure" large bbox, GT is small central area

**High-confidence mismatch (2 cases):**
- `20210910_204401.jpg`: "framed artwork" at 0.99 confidence in corner beats statues
- `DSC_0001_BURST20241121142123881.JPG`: "painted figure" (small) beats "painting" (correct larger area) — REGRESSION from >50% penalty

**Class tier / scoring edge (2 cases):**
- `DSC_4291.JPG`: "mosaic" selected, GT wants "figure figne" (garbled class, gets default 1.5x)
- `DSC_0155.JPG`: "painted figure" wins over GT vase area — REGRESSION

**Attempted: containment/nesting logic (2026-02-15) — REJECTED**
Tried preferring the smaller detection when it's contained within a larger same-tier one (≥80% overlap, ≥8x area ratio, winner ≥15% of image). Regressed accuracy from 72.4% to 58.6%. Root cause: most large specific_art detections (mural, framed artwork) ARE the subject and naturally contain smaller detail detections (figurine, mosaic) within them. Only 1 of ~18 triggered swaps was correct (DSC_4385). No threshold combination could separate signal from noise.

### P1: Detection miss / tiny subject (5 of 22 bad)
Subject not detected or too small for models to identify:
- `DSC_0493.JPG`: chalk drawing of woman's face → "artistic object" wins
- `DSC_4162.JPG`: tiny pig face mosaic at bottom of archway → "decorated sign" wins
- `DSC_4311.JPG`: rocket tile art above street sign → wrong area selected
- `DSC_4312.JPG`: small decorative tile → "mosaic" selected at wrong position
- `DSC_4388.JPG`: hard-to-see mosaic aliens → "framed artwork" wins

### P1: Incomplete / partial bbox detection (3 of 22 bad)
- `20210626_160627.jpg`: mural detection shifted, doesn't cover the actual ant statue
- `DSC_3367.JPG`: painted figures on pillar only partially detected (bottom portion)
- `DSC_4059.JPG`: full PARIS mural not detected as single entity, only fragments

### P1: Sign vs art disambiguation (2 of 22 bad)
- `DSC_4042.JPG`: "painting" misdetecting a decorative street sign, GT wants "decorative art"
- `DSC_4381.JPG`: "decorated sign" (higher conf, larger) beats "sculpture"
- Note: `DSC_0274.JPG` "decorated sign" IS correct (it's a tile mural) but "painted figure" wins instead — overlaps with wrong-primary

### P1: Class scoring edge case (1 of 22 bad)
- `20210815_163856.jpg`: "art installation exhibit" still selected despite removal from specific_art_classes. Falls to scene_art 2.0x but no better detection for the small GT manual box area.

## Filtering

### P1: Non-art image filtering
Auto-detect and skip images that aren't interesting art subjects (9 "other" include 4 not-art):
- Museum exhibit labels / info cards (`DSC_4063.JPG`, `DSC_4074.JPG`)
- Photos of cars, buildings without art (`WP_20170624_15_17_38_Pro.jpg`)
- Images dominated by glass reflections (`20210910_203723.jpg`, `DSC_3065.JPG`)

6 images already marked `"not_art": true` in ground truth. Pipeline should skip these.

## Feature Requests

### P2: Smart sub-crop / focal area refinement
When detection is correct but covers a large area, zoom into the most interesting detail (face, focal point). 3 images rated "other":
- `DSC_0153.JPG`: focus on woman's face within mural
- `DSC_1488.JPG`: focus on woman's face within artwork
- `DSC_3065.JPG`: focus on central area (also glass reflection issue)

### P3: Multi-crop for panoramic scenes
Generate multiple output crops from a single image when it contains multiple distinct art pieces.
- `DSC_3089.JPG`: 3 columns with murals + 1 lower panel (4 manual boxes in GT)

### P3: Multiple valid detections
- `DSC_4205.JPG`: mural selected, but GT has 3 valid detections (statue, painted figure, figurine). Could benefit from multi-crop or user selection.
