# Backlog

## Session Summary (2026-03-15)

### Visual review: 122 images reviewed by Sonnet agents, 2 cropper improvements

**Visual review process:** 9 parallel Sonnet agents reviewed all 122 images (scene + individual crop files). Found 41 findings across 6 categories: 15 bad secondary/focal crops, 6 non-art content selected, 4 signage intrusion, 7 known IoU misses, 4 missed opportunities, 5 minor framing.

**Improvement 1: Text filter for inner focal crops**
Added `text_ratio` check to `_get_quality_inner_detections` acceptance loop. Previously, text filtering only applied to secondary crops — inner focal crops (within a large primary) skipped the check. Fixes edge cases where text-heavy regions (signs, placards) become focal crop targets.

**Improvement 2: Dual confidence threshold for inner focal candidates**
Ensemble-pass detections reused as inner focal anchors now require higher confidence (0.35 vs 0.25 for focal-pass detections). This fixes the Pichiavo placard issue (20210808_162451) where a "mosaic@0.254" detection from the ensemble pass was selected as an inner focal crop — it was actually an information placard mounted on the mural. Focal-pass detections (face/figure/head prompts) keep the lower 0.25 bar since they're purpose-built for finding interesting art details.

**Confirmed fix:** 20210808_162451 went from 4 crops (including placard) to 3 crops (mural + 2 face/figure details). DSC_0154 inner focal selection improved (ensemble "painted figure" replaced by focal-pass detection). IoU hit rate unchanged at 115/122 (94%).

**Improvement 3: Person-as-art filter (COCO person detector)**
Added `detect_persons()` to `OptimizedEnsembleDetector` using yolov8n (COCO-trained). When the primary is a "figure" class, checks if a COCO person detection (conf>0.5) overlaps it with IoU>0.5. If so, the "figure" is likely a real person, not art — suppress and re-select.

Fixed: DSC_0001_BURST (IoU miss → hit). Person detected at conf=0.87 with IoU=0.91 overlap. After suppression, "painting" becomes primary and better covers the GT installation area. False positive rate: near-zero — COCO person detectors don't fire on painted figures in murals (tested on 80+ images).

**IoU hit rate: 115/122 (94%) → 116/122 (95%) → 117/122 (96%)**

**Improvement 4: VLM vote-based confidence (replaces flat 0.80)**
Small VLMs (Qwen3-VL-2B) often return many near-duplicate boxes for the most
prominent subject — 44% of responses have 3+ boxes, some hit 30. This
autoregressive "hedging" is a genuine confidence signal: the more often the
model outputs overlapping boxes, the more salient the subject.

Changed `_vlm_boxes_to_detections()` to cluster raw VLM boxes by IoU>0.3
and set confidence proportional to cluster vote share (0.40 base + 0.55 ×
vote_ratio). Dominant clusters get ~0.90+ while singleton spurious boxes
get ~0.43. No cache invalidation needed — pure post-processing.

Fixed: DSC_4385 (mosaic IoU miss → hit). VLM returned 29 boxes: 27 cluster
around the correct Space Invader mosaic (conf=0.91), 2 on the wrong wall
region (conf=0.43). Previously all had flat 0.80 and size_bonus decided.

**Detection accuracy explorations (2026-03-15):**
- **Learned reranker** (sklearn LogReg/RF/GBT): FAILED — every model performed worse than heuristic (-1 to -2). Dataset too small (122 images), heuristic encodes domain knowledge scalar features can't capture.
- **Scoring gap analysis**: DSC_4385 root cause is VLM flat confidence (all boxes get 0.80). Size_bonus (1.20 vs 0.70) kills the correct small mosaic. DSC_4166 VLM points at wrong object entirely.
- **VLM confidence options**: logprobs from llama-server (promising), prompt-based confidence field (risky, invalidates cache), multi-pass consensus (slow but robust).

**Remaining unfixable issues (detector limitations):**
- Vandalism tags classified as "mural"/"street art" (20210530 C2, DSC_4414 C2, DSC_4303 C2) — models can't distinguish art graffiti from tags
- Building pipes classified as "art installation"/"sculpture statue" (DSC_3401 C2/C3) — Centre Pompidou pipes genuinely resemble sculptures to the model
- People detected as "painted figure" (DSC_2149 C2, DSC_0001_BURST C1) — visual similarity to art
- EasyOCR at 320px can't read French text or stylized graffiti (Pichiavo placard text_ratio=0.016)

---

## Session Summary (2026-03-14)

### IoU hit rate: 112/122 (92%) → 115/122 (94%)

**VLM prompt improvement (net +3):**
Rewrote Qwen3-VL grounding prompt to focus on "prominent visual artworks" and exclude graffiti tags, shop window objects, and signage. Tested 5 prompt variants (V2-V5, V4b, V4c) on representative subsets before full eval.

| Variant | Key change | Result |
|---------|-----------|--------|
| V2 | "ONLY on facades/walls" | 113/122 (+1), regresses DSC_4305 |
| V3 | Just shop window exclusion | No effect |
| V4 | "Prominent artworks, not tags/scrawls, not through glass" | **115/122 (+3)** |
| V5 | "Single most prominent" | Too aggressive, loses multi-art |
| V4b | V4 without tag exclusion | DSC_4399 regression persists |
| V4c | Current + "prominent" + shop window | DSC_4414 graffiti flood returns |

V4 chosen: fixes DSC_4302 (shop window), DSC_4414 (graffiti flood 29→2 boxes), DSC_3401 (pipes), 20220219. Regresses DSC_4291 (borderline small sculpture on ornate facade).

**Other improvements:**
- Removed unused COCO `_art_related_classes` (zero appearances across 122 images)
- Removed `vase` from art classes (only appeared as misdetected plant box)
- Added relative brightness filter for secondary crops (filters shadow/dark regions)
- Defaulted `--vlm` and `--multi-crop` to on in CLI
- Removed HuggingFace transformers VLM fallback (llama-server only)
- Audited all scoring heuristics — all active, no dead code remaining

**EAST text detection filter:**
Added `TextDetector` using OpenCV EAST model (~93MB, ~100ms/crop at 320px) to filter text-heavy regions. Applied at two levels:
- **Secondary crop filter**: skip secondaries with >10% text ratio (fixes DSC_3401 pipe, DSC_4414 tagged door)
- **Primary re-selection**: if primary has >10% text, remove it and re-select from remaining detections (fixes DSC_4166 pigeon sculpture, 20210424 mural, DSC_4201 cartoon figures, DSC_3167 graffiti text, DSC_4063 gallery label)

Switched from EAST to EasyOCR (CRAFT) `readtext()` after evaluation:
- Fewer false positives on art (DSC_0153: 0.7% vs EAST's 7.9%)
- Catches graffiti text EAST missed (DSC_3167)
- `readtext()` eliminates geometric art false positives (DSC_4158 was falsely
  filtered by `detect()` which read the mural's lines as text)
- Short/low-confidence OCR results filtered (<=3 chars at conf<0.5, or conf<0.1)
- Polygon area via Shoelace formula (axis-aligned bbox inflated rotated text areas)
- Versioned cache (`cache/text_detect/`) — version bumped on filter/area changes
- `torch.compile` for ~2x CPU speedup
- French language support for Parisian signs

**Remaining limitations:**
- DSC_4074 (museum labels): text too small at 320px
- 20210808 (info panel on mural): text too small at 320px
- EasyOCR is slightly non-deterministic — results can vary between runs
  (mitigated by caching; once cached, results are stable)

### Saliency-guided focal point for wide primaries — NOT NEEDED

Investigated using saliency analysis within wide primary bboxes for better crop anchors. Installed `opencv-contrib-python`, implemented, tested on all 122 images.

**Result:** No impact. The existing focal detection pass (DINO face/figure prompts) already handles all 6 target images. Reverted.

See Investigation Roadmap item #1 for full details.

---

## Session Summary (2026-03-04) — resolution experiment

### 1024px VLM resolution experiment — net zero
Tested `--vlm-max-image-size 1024` (vs default 512px) across all 122 images.

| Class | 512px IoU | 1024px IoU | Δ |
|-------|-----------|------------|---|
| mural | 38/38 (100%) | 37/38 (97%) | -1 |
| mosaic | 21/25 (84%) | 22/25 (88%) | +1 |
| street_art | 20/22 (91%) | 19/22 (86%) | -1 |
| sculpture | 18/20 (90%) | 19/20 (95%) | +1 |
| **TOTAL** | **112/122 (92%)** | **112/122 (92%)** | **0** |

**Conclusion**: net zero overall. 1024px finds more mosaics but loses a mural and street art image. At ~4x slower inference per triggered image, 512px remains the better default.
Root cause for DSC_4302 (mosaic): VLM at 1024px finds art fragments visible through a store window rather than the central mosaic — detection problem, not a resolution problem.

### Infrastructure fix
- `scripts/evaluate_art_class_accuracy.py`: `--vlm-gguf` and `--vlm-mmproj` now default to `models/qwen3vl/` paths (same as CLI), so `--vlm` alone uses llama-server correctly without passing explicit paths.

---

## Session Summary (2026-03-04) — continued

**Current accuracy: 112/122 (92%) IoU hit rate** with `--vlm` (llama-server GGUF, ~20s/image, cached after first run). Baseline without VLM: 107/122 (88%).

### VLM integration milestones this session
| Version | IoU | Change |
|---------|-----|--------|
| Baseline (YOLO/DINO only) | 107/122 (88%) | — |
| VLM v1 (heuristics A+C, generic prompt) | 93/122 (76%) | −14 regression |
| VLM v2 (heuristic-C only, specific prompt) | 110/122 (90%) | +3 |
| VLM v3 (+ heuristic-D conf<0.35, size_bonus floor) | 112/122 (92%) | +2 |
| VLM v4 (+ merge source preservation) | 112/122 (92%) | swap: DSC_4312↑, DSC_4166↓ |

### Remaining 7 IoU misses (updated 2026-03-14, after V4 prompt)
- **Borderline (1)**: 20200525 (0.11) — mosaic det, box slightly off
- **Installation/person-as-art (2)**: 20210910_203723, DSC_0001_BURST — "painted figure" wins
- **Hard mosaics (2)**: DSC_4162 (tiny pig, <0.2% area), DSC_4385 (scoring gap)
- **Wrong-region (1)**: DSC_4166 — correct class, bbox in wrong part of image
- **VLM prompt regression (1)**: DSC_4291 — V4 prompt finds "painting" on ornate facade instead of small sculpture

Previously fixed by V4 prompt (2026-03-14): ~~DSC_4302~~, ~~DSC_4414~~, ~~DSC_3401~~, ~~20220219~~

### Infrastructure improvements
- `--vlm` alone now sufficient: GGUF paths default to `models/qwen3vl/`; `--vlm-gguf`/`--vlm-mmproj` only needed to override
- `scripts/download_models.py --vlm` downloads both GGUFs from HuggingFace (~2.6 GB) with llama-server build instructions
- README/USAGE docs updated: VLM feature, setup, 92% accuracy figure

---

## Session Summary (2026-03-04)

**Accuracy: 107/122 (88%) IoU hit rate** (YOLO/DINO ensemble baseline).
Qwen3-VL-2B standalone eval: 14/17 previously-failing images now found → **theoretical ceiling 121/122 (99%)** if integrated as fallback.

### Completed this session
- Evaluated grounding-dino-base: net **-12** vs tiny (78% vs 88%). Larger backbone doesn't help; generic labels hurt primary selection. Keep tiny.
- Evaluated YOLOE-26m: net **-7** vs YOLO-World (82% vs 88%). Class calibration worse (smears murals → street_art). Keep YOLO-World.
- Implemented `scripts/evaluate_qwen3vl.py`: grounding + VQA eval on 17 miss images. Supports Qwen3-VL-2B and Qwen3.5-2B.
- Evaluated **Qwen3-VL-2B** (`Qwen/Qwen3-VL-2B-Instruct`): **14/17 hits** on previously failing images. Finds Space Invader mosaics, sculptures, installations that YOLO/DINO miss entirely. VQA 2/4 category C correct.
- Evaluated **Qwen3.5-2B** (`Qwen/Qwen3.5-2B`): **13/17 hits**. Slightly worse. Architecture mismatch warnings (loaded via `Qwen3_5ForConditionalGeneration` fallback). transformers upgraded to 5.3.0.dev0.
- **Winner: Qwen3-VL-2B** — purpose-built for grounding, no arch mismatch, 1 more hit.
- Persistent misses for both VLMs: DSC_4311, DSC_4388 (tiny pixel-art Space Invader mosaics, sub-1% image area).
- Cache: `cache/qwen3vl/`, keyed by model_id + prompt_mode + max_image_size.
- CPU inference at 512px: ~5-10 min/image. GPU strongly recommended for pipeline use.
- **Integrated Qwen3-VL into pipeline** (`--vlm` / `--vlm-confirm` / `--vlm-max-image-size` CLI flags):
  - Pass 3 added to `OptimizedEnsembleDetector.detect()` — fires after pass 1+2 and merge
  - Cache key identical to `evaluate_qwen3vl.py` → all 17-miss eval results reused instantly
  - `--vlm`: fallback mode (fires when no viable central candidate found)
  - `--vlm-confirm`: confirmation mode (runs on every image)
  - Baseline `--vlm` run: VLM triggered on **24/122** images; 98 skipped with viable candidate
- **Report enhancements for VLM visibility**:
  - VLM boxes rendered in cyan; primary color indicates source (green=YOLO, chartreuse=DINO, yellow=MIXED, cyan=VLM)
  - VLM badge + count in image info panel; VLM filter button
  - Config summary panel shows VLM mode, model, and max image size
  - `Detection.source` field tracks `"yolo"` / `"dino"` / `"vlm"` through merge pipeline
  - `_last_vlm_detections` on detector stores raw pre-merge VLM boxes for accurate count (source tags lost during merge for YOLO/DINO overlap)
- **Expanded VLM heuristics A+C** added to pass-3 trigger (beyond no-candidate fallback):
  - **A** (weak art score): fires when `primary.confidence × class_multiplier < 2.0` — targets `art_installation` / low-conf generic detections winning primary
  - **C** (close competition): fires when top-2 simplified scores within 20% ratio — targets coin-flip tie scenarios (DSC_4385 etc.)
  - Option B (non-specific-art primary) kept as future option if A+C insufficient
  - Verbose output shows trigger reason: e.g. `VLM/heuristic-A (art_score=0.70<2.0)`

### What Qwen3-VL fixes vs YOLO/DINO
| Category | Images | YOLO/DINO | Qwen3-VL |
|----------|--------|-----------|----------|
| B: wrong detection wins | DSC_4020, DSC_4371, DSC_4385, DSC_3401 | miss | HIT |
| D: not detected at all | 20210911, DSC_4168, DSC_4291, DSC_4312 | miss | HIT |
| A: borderline box | DSC_4101_0, DSC_4388 | miss | HIT / miss |
| C: person-as-art (VQA) | 20210910_191256, _204401 | n/a | correct |

### VLM integration results — v2 (heuristic-C only + new prompt)

**110/122 (90%) IoU hit rate** — up from 107/122 (88%) baseline. **+3 improvement.**
Heuristic-A disabled. Heuristic-C (top-2 within 20% ratio) kept.
New prompt: removed "artwork", added "fresco"/"statue"/"graffiti"; explicit exclusion of street signs/exhibit labels/shop signs/decorative typography.
Prompt hash in cache key → auto-invalidates on prompt changes.

| Metric | Baseline | VLM v1 (A+C) | VLM v2 (C-only) |
|--------|----------|--------------|-----------------|
| IoU hit rate | 107/122 (88%) | 93/122 (76%) | **110/122 (90%)** |
| User feedback | 87.7% | 89.3% | TBD |

Per-class IoU: mural 38/38 (100%), mosaic **19/25 (76%)** +3, street_art 20/22 (91%), sculpture 18/20 (90%), painting 9/9 (100%), installation 2/4 (50%), non_art 4/4 (100%).

### VLM integration results — v1 (heuristics A+C, first run)

93/122 (76%) IoU — regression from baseline. Heuristic-A fires on 53 images, VLM "artwork" overrides correct YOLO/DINO detections. DSC_4042, DSC_4168 regressed. User feedback improved to 89.3% because visual regions are better even when labeled generically.

---

## Session Summary (2026-02-26)

**Accuracy: 103/122 (84%) → 107/122 (88%)** (class audit + scoring improvements).
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

## Remaining Issues (audited 2026-03-14)

*Re-audited against current detection. Of original 30 feedback items, 13 now fixed without VLM.*
*With `--vlm` (112/122 baseline), only 10 total IoU misses remain across the full dataset.*

### Confirmed MISS with --vlm (4 backlog items in the 10 VLM misses)

These fail both with and without VLM — the hardest cases:
- `DSC_4385.JPG` (IoU=0.01): mosaic→graffiti, scoring gap
- `DSC_4162.JPG` (IoU=0.00): tiny pig face mosaic, sculpture wins
- `DSC_3401.JPG` (IoU=0.00): mosaic→sculpture, wrong detection wins
- `DSC_0001_BURST` (IoU=0.00): person walking→painted figure

### Confirmed FIXED by VLM (14 backlog items, verified 2026-03-14)

These MISS without `--vlm` but are confirmed HIT with `--vlm`:
- `DSC_4388.JPG` (IoU 0.16→0.61): sculpture wins with VLM
- `DSC_0155.JPG` (IoU 0.00→0.94): painting wins with VLM
- `DSC_4291.JPG` (IoU 0.02→0.39): sculpture wins with VLM
- `DSC_0274.JPG` (IoU 0.15→0.85): painting wins with VLM
- `DSC_3367.JPG` (IoU 0.22→0.80): graffiti wins with VLM
- `DSC_4371.JPG` (IoU 0.01→0.63): street art wins with VLM
- `DSC_4059.JPG` (IoU 0.17→0.91): street art wins with VLM
- `20210911_152658.jpg` (IoU 0.00→0.52): street art wins with VLM
- `DSC_0493.JPG` (IoU 0.01→0.69): painting wins with VLM
- `DSC_4311.JPG` (IoU 0.00→0.55): sculpture wins with VLM
- `DSC_4312.JPG` (IoU 0.00→0.81): street art wins with VLM
- `DSC_3065.JPG` (IoU 0.09→0.85): art installation wins with VLM
- `DSC_4205.JPG` (IoU 0.05→0.45): decorative art wins with VLM
- `DSC_1045.JPG` (IoU 0.30→0.30): figurine, borderline either way

### FIXED since original feedback (now HIT without --vlm)

- ~~`DSC_4042.JPG`~~ (IoU=0.97): "decorative art" now wins correctly
- ~~`DSC_4381.JPG`~~ (IoU=0.97): "sculpture" now wins correctly
- ~~`20200525_170722.jpg`~~ (IoU=0.71): sculpture/mosaic now selected correctly
- ~~`DSC_4201.JPG`~~ (IoU=0.52): mural now detected
- ~~`DSC_4294.JPG`~~ (IoU=0.99): "art installation" now wins correctly
- ~~`20210815_163856.jpg`~~ (IoU=0.73): correct primary now selected
- ~~`20210910_204401.jpg`~~ (IoU=0.99): fixed by edge penalty (known)
- ~~`DSC_0153.JPG`~~ (IoU=1.00): focal detection finds face correctly
- ~~`DSC_1488.JPG`~~ (IoU=0.64): focal detection finds face
- ~~`20210530_135908.jpg`~~ (IoU=0.99): mural detected correctly
- ~~`20210808_162451.jpg`~~: mural detected (no GT to measure)
- ~~`DSC_4168.JPG`~~ (IoU=0.98): correct detection now
- ~~`DSC_4305.JPG`~~ (IoU=0.98): correct detection now

### ~~Crop-only issues~~ — ALL FIXED (2026-03-14)

- ~~`20210213_154948.jpg`~~: dark "painted figure" (building behind fence) filtered by relative brightness check
- ~~`DSC_0312.JPG`~~: "vase" (plant box) filtered after removing vase from art classes
- ~~`DSC_3614.JPG`~~, ~~`DSC_4382.JPG`~~, ~~`DSC_4399.JPG`~~: secondary quality filters already blocked these

---

## TODO — Testing

### Update and expand unit tests

Tests are stale — many don't cover recent changes. Key areas to add/update:
- **TextDetector**: polygon area computation (Shoelace), confidence filtering, cache versioning
- **Pipeline**: `run_detection_pipeline()` text-heavy primary re-selection
- **Cropper**: secondary text/brightness filters
- **Scoring**: current class multiplier tiers (COCO classes removed, vase removed)

### Add text detection visualization to report

Show which bboxes were text-filtered in the detection overlay (e.g., dashed outline or different color) so regressions are immediately visible.

---

## TODO — Refactoring

### ~~Deduplicate report and preprocessor detection pipelines~~ — DONE (2026-03-14)

Extracted to `pipeline.py`.

### Deduplicate remaining report helpers

`report.py` and `preprocessor.py` duplicate detection → primary selection → text filtering → focal detection → cropping logic. This leads to behavior divergence and maintenance burden. Extract shared pipeline into a single function that both call.

**Files:** `src/frame_prep/report.py` (lines ~166-220), `src/frame_prep/preprocessor.py` (lines ~127-190)

---

## Investigation Roadmap

### 1. ~~Saliency-guided focal point for wide primaries~~ — INVESTIGATED, NOT NEEDED (2026-03-14)

**Finding:** The 6 "wide primary bad crop" images (DSC_0153, DSC_1488, 20210530_135908, DSC_4205, 20210808_162451, DSC_1045) are **already handled by the existing focal detection pass** — DINO face/figure prompts find quality inner detections for all of them. The report confirms focal anchor points (faces) are shown correctly.

**What was tried:**
- Installed `opencv-contrib-python` to enable `cv2.saliency.StaticSaliencySpectralResidual`
- Added saliency fallback in `crop_with_detections()` for when primary fills frame and no inner dets exist
- Improved `find_interest_points()` to use Gaussian-smoothed weighted centroid instead of raw argmax (which lands on noisy edge pixels)

**Why it had no effect:**
- Only 2/122 images in the dataset have fills-frame + no-focal-detections (DSC_3167, DSC_4074) — both are `non_art` ground truth misdetections
- `StaticSaliencySpectralResidual` responds to frequency-domain anomalies, not semantic content — the centroid lands near the mural center anyway, not on faces/figures
- The existing focal detection pass (DINO with face/figure prompts) already solves the problem these 6 images were listed for

**Conclusion:** Reverted. The remaining crop quality issues for these images are about *which* focal point is best, not about having no focal point. DSC_1045 is actually a detection issue (wrong figurine selected, 16.5% area, not a wide primary). 20210808_162451 has no GT annotation.

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

**Florence-2 REGION_TO_DESCRIPTION assessed (2026-02-26) — NOT PURSUED:**
- Explored loading Florence-2-base on transformers 5.0.0; requires 3 HF cache patches and CPU inference at 5–15s/detection.
- Only 2 target images in category C; maximum gain +2 = 109/122 (89.3%).
- Decision: cost/complexity not justified. Same fundamental limitation as SigLIP: distinguishing a painted figure from a real person is hard even with region conditioning when the crop content is visually ambiguous.
- Revisit if: GPU available, or a fast API-based VLM endpoint emerges (DINO-X native region QA, GPT-4V, etc.).

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
- **Qwen3-VL VQA** (EVALUATED 2026-03-04): 2/4 correct on category C. Correctly identifies sculpted/depicted figures; the 2 "real person" answers may be correct (GT labels are "installation", not person-in-art). This is the best approach tried so far.

**For small mosaic detection (items 3 + bad_detection):**
- **OWLv2**: EVALUATED (2026-02-26) — does not beat baseline. Best F1=0.293 (art_only prompts) vs baseline F1=0.529. Not worth integrating.
- **Qwen3-VL grounding** (EVALUATED 2026-03-04): 14/17 hits on previously-failing images. **Most promising path** — integrate as fallback pass in pipeline when YOLO/DINO has no viable candidate.

**Pipeline integration of Qwen3-VL (next step):**
- Add `_run_qwen_vlm(image, ...)` method to `OptimizedEnsembleDetector`
- Add `use_vlm: bool = False` constructor param + `--vlm` CLI flag
- Trigger VLM pass when pass 1+2 have no viable central candidate
- Convert 0-1000 normalized bbox → pixel `Detection` with conf=0.8, class_name from label
- Cache in `cache/qwen3vl/` using same hash scheme as `evaluate_qwen3vl.py`
- Note: CPU inference 5-10 min/image → GPU strongly recommended; cached results reused instantly

**For full detection upgrade:**
- **grounding-dino-base** (IDEA-Research): EVALUATED (2026-03-02) — **does not beat tiny**. IoU hit rate 95/122 (78%) vs tiny 107/122 (88%) — net **-12**. Class accuracy identical (66/122). Base produces more generic labels ("painting artwork", "framed artwork") that win primary selection but cover wrong regions. Mural -4, mosaic -3, non_art -3. Root cause: larger backbone doesn't help when the bottleneck is open-vocab label calibration, not backbone capacity. Stick with tiny.
- **YOLOE-26m** (Ultralytics, 2025): EVALUATED (2026-03-02) — **does not beat YOLO-World**. IoU 100/122 (82%) vs baseline 107/122 (88%) — net **-7**. Class accuracy 50/122 (41%) vs 66/122 (54%). Root cause: smears murals into "street art" (23/38 murals classified as street_art), causing primary-selection losses. Bboxes are fine but class calibration between mural/street_art worse than YOLO-World. Stick with yolov8m-worldv2.
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
