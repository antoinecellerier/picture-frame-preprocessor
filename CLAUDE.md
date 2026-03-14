# Picture Frame Preprocessor

Art photo preprocessor for e-ink picture frames. Detects art subjects in museum/gallery photos using YOLO-World + Grounding DINO ensemble, crops and zooms to highlight them. Optional Qwen3-VL VLM fallback.

## Environment

- **Always use `venv/bin/python`** — system `python` is not available
- CPU-only machine (Intel i7-1270P, AVX2, no discrete GPU, 30GB RAM)
- Always resize images before VLM inference (`--vlm-max-image-size 512` default). Never run VLM at native resolution (3000-4000px) — it takes hours per image on CPU.
- VLM uses **llama-server** (`~/stuff/llama.cpp/build/bin/llama-server`), NOT PyTorch/transformers. GGUF models in `models/qwen3vl/`. The `--vlm` flag alone is sufficient.
- pytest: `venv/bin/python -m pytest tests/ -x -q`

## Evaluation & regression testing

**After ANY change to detector.py, cropper.py, or scoring logic**, run the eval and compare to baseline:

```bash
venv/bin/python scripts/evaluate_art_class_accuracy.py --vlm | tee /tmp/eval_out.txt
```

The current baseline is stored in `eval-baseline.json`. Compare IoU hit rate before committing. If results regress, revert — don't commit net-negative changes.

Use `/eval` to run evaluation with automatic baseline comparison.

## Report generation

```bash
venv/bin/python -m frame_prep.cli report
```

Output: `reports/interactive_detection_report.html`. Use `/report` to generate in background.

## Key conventions

- **Current detector class**: `OptimizedEnsembleDetector` (not `ArtFeatureDetector` — that's been removed)
- **Current cropper class**: `SmartCropper`
- **Text detection**: `TextDetector` in `analyzer.py` uses EAST model (`models/frozen_east_text_detection.pb`) at 320px to filter text-heavy regions (signs, labels). Threshold: >10% text ratio.
- CLI entry point: `frame_prep.cli` (subcommands: `process`, `batch`, `report`)
- CLI defaults: `--vlm` and `--multi-crop` are on by default; use `--no-vlm` / `--no-multi-crop` to disable
- Test dataset: `test_real_images/` (122 images with ground truth annotations)

## Experimental features

- **Validate effectiveness BEFORE full integration** — run eval on a prototype/branch first. Don't build out a full feature only to discover it's net-negative.
- Default new detectors/models to **opt-in** (behind a flag) until proven net-positive on the 122-image test set.
- Document significant findings in `BACKLOG.md`, not only in MEMORY.md.

## Improving Claude instructions

This file, the slash commands in `.claude/commands/`, and the memory files are all living documents. When you notice recurring friction, repeated corrections, or workflows that could be automated:

- **Update CLAUDE.md** with new rules or conventions learned from the session
- **Create or update slash commands** in `.claude/commands/` for repeated multi-step workflows
- **Update memory files** with findings that should persist across sessions (see MEMORY.md for the index)
- **Update `eval-baseline.json`** when eval results improve (the `/eval` skill handles this)

Proactively suggest improvements at the end of a session if patterns emerged.

## Committing

- **Update docs before committing** — update BACKLOG.md, MEMORY.md, and CLAUDE.md with relevant findings before creating the commit, and include them in the same commit.

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
