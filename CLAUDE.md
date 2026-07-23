# Picture Frame Preprocessor

Art photo preprocessor for e-ink picture frames. Detects art subjects in museum/gallery photos using YOLO-World + Grounding DINO ensemble, crops and zooms to highlight them. Optional Qwen3-VL VLM fallback.

## Environment

- **Always use `venv/bin/python`** — system `python` is not available
- venv is Python 3.14 with **CPU-only torch wheels** (`--index-url https://download.pytorch.org/whl/cpu`). When reinstalling torch, keep it CPU-only — default wheels add ~5GB of unused CUDA libraries.
- CPU-only machine (Intel i7-1270P, AVX2, no discrete GPU, 30GB RAM)
- Always resize images before VLM inference (`--vlm-max-image-size 512` default). Never run VLM at native resolution (3000-4000px) — it takes hours per image on CPU.
- VLM uses **llama-server** (`~/stuff/llama.cpp/build/bin/llama-server`), NOT PyTorch/transformers. GGUF models in `models/qwen3vl/`. The `--vlm` flag alone is sufficient.
- pytest: `venv/bin/python -m pytest tests/ -x -q`

## Evaluation & regression testing

**After ANY change to detector.py, cropper.py, pipeline.py, analyzer.py, or scoring logic**, run the eval and compare to baseline:

```bash
venv/bin/python scripts/evaluate_art_class_accuracy.py --vlm | tee /tmp/eval_out.txt
```

The eval uses `run_detection_pipeline()` — the same pipeline as the real tool. This ensures the eval measures what users actually see (including text filtering, person filtering, etc.). **Never bypass the pipeline in the eval** — discrepancies between eval and production cause inflated numbers.

The current baseline is stored in `eval-baseline.json`. Compare IoU hit rate before committing. If results regress, revert — don't commit net-negative changes.

Use `/eval` to run evaluation with automatic baseline comparison.

The IoU threshold (`IOU_THRESHOLD = 0.15`) is defined in `defaults.py` and shared by both the eval script and the report. Don't hardcode thresholds elsewhere.

## Report generation

```bash
venv/bin/python -m frame_prep.cli report
```

Output: `reports/interactive_detection_report.html`. Use `/report` to generate in background.

## Sample images for README

```bash
venv/bin/python scripts/create_sample_composites.py
```

Regenerates all sample composites and the report screenshot using the current pipeline. Outputs to `samples/`. Uses playwright for the screenshot — requires `venv/bin/playwright install chromium`.

## Key conventions

- **Current detector class**: `OptimizedEnsembleDetector` (not `ArtFeatureDetector` — that's been removed)
- **Current cropper class**: `SmartCropper`
- **Text detection**: `TextDetector` in `analyzer.py` uses EasyOCR `readtext()` at 320px to filter text-heavy regions (signs, labels). Uses `center_weighted_text_ratio` (text near bbox center counts fully, edge text discounted). Threshold: `TEXT_RATIO_THRESHOLD = 0.20` in `analyzer.py`. Versioned cache in `cache/text_detect/` (v6). Uses Shoelace formula for polygon area, filters short/low-conf OCR results and single-char false positives on art textures.
- **Person filter**: `detect_persons()` in detector.py uses YOLOv8n (COCO class 0) to suppress "painted figure" detections that are actually real people. Pipeline step 3b.
- **VLM confidence**: `_vlm_boxes_to_detections()` clusters duplicate VLM boxes by IoU>0.3 and uses vote count as confidence (not flat 0.80). Small orphan VLM boxes that don't overlap YOLO/DINO detections are suppressed via proximity check (near-orphan < 8% diagonal = suppress, far-orphan = keep).
- CLI entry point: `frame_prep.cli` (subcommands: `process`, `batch`, `report`)
- **Defaults**: All defaults live in `defaults.py` as the single source of truth. CLI flags, batch workers, sample scripts, and `create_detector()` all reference it. Don't hardcode default values elsewhere.
- CLI defaults: `--vlm` and `--multi-crop` are on by default; use `--no-vlm` / `--no-multi-crop` to disable
- **Pipeline**: `pipeline.py` contains the shared detection pipeline (`run_detection_pipeline()`). Both `preprocessor.py` and `report.py` call it — don't duplicate detection logic.
- Test dataset: `test_real_images/` (122 images with ground truth annotations)

## Experimental features

- **Validate effectiveness BEFORE full integration** — run eval on a prototype/branch first. Don't build out a full feature only to discover it's net-negative.
- Default new detectors/models to **opt-in** (behind a flag) until proven net-positive on the 122-image test set.
- Document significant findings in `BACKLOG.md`, not only in MEMORY.md.
- **Avoid overfitting to the eval dataset** — tuning a threshold by 0.001 to catch one specific image is not a real improvement. Prefer structural fixes (new signals, new filtering approaches) over fragile parameter tweaks.
- **Local-only** — the user wants the tool to run entirely locally. Don't propose API-based solutions (Claude API, cloud services). CPU-compatible models only.

## Improving Claude instructions

This file, the slash commands in `.claude/commands/`, and the memory files are all living documents. When you notice recurring friction, repeated corrections, or workflows that could be automated:

- **Update CLAUDE.md** with new rules or conventions learned from the session
- **Create or update slash commands** in `.claude/commands/` for repeated multi-step workflows
- **Update memory files** with findings that should persist across sessions (see MEMORY.md for the index)
- **Update `eval-baseline.json`** when eval results improve (the `/eval` skill handles this)

Proactively suggest improvements at the end of a session if patterns emerged.

## Committing

- **Update docs before committing** — update BACKLOG.md, MEMORY.md, and CLAUDE.md with relevant findings before creating the commit, and include them in the same commit.
- **Update README.md** when user-facing behavior changes (accuracy, features, CLI defaults).
- **Regenerate samples** when detection or crop behavior changes: `venv/bin/python scripts/create_sample_composites.py`. Include updated samples in the commit.

## Running commands

- **Always `tee` output** to a `/tmp` file when running eval, detection, or any command you may need to search afterwards (e.g., `| tee /tmp/eval_out.txt`). Then use Grep/Read on the file instead of re-running the command with different pipes.
- **Use the Write tool** to create temp scripts at `/tmp/*.py` paths instead of heredocs (`cat << 'EOF'`) or long inline `python -c` strings — those trigger permission prompts.
- Run temp scripts with a simple `venv/bin/python /tmp/script.py` bash call.

## What NOT to do

- Don't use PyTorch/transformers for VLM inference — the project uses llama.cpp GGUF
- Don't run images at native resolution through VLM on this CPU
- Don't add YOLO-World prompts without evaluating — the 28-class list is zero-sum; new classes steal attention from existing ones
- Don't commit scoring/detection changes without running `/eval` first
- Don't re-run expensive commands just to grep different parts of the output — tee first, then search the file
- Don't measure eval results differently from the real pipeline — if the eval bypasses text filtering or other pipeline steps, the numbers are inflated and misleading
- Don't propose cloud/API-based detection solutions — the tool must run fully local
