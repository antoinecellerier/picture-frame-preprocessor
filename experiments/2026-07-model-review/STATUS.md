# July 2026 model review — experiment status

Master plan: see "Evaluate new detection-quality and speed options (July 2026 review)"
(plan file from the 2026-07-23 session; key constraints reproduced here).

**Baseline: 116/122 (95.1%) IoU with `--vlm`** (`eval-baseline.json`, 2026-03-15). `master` stays at this baseline; experiments run on `exp/*` branches. Only net-positive changes merge.

## Standing rules (every session)

1. Read this file first; check `logs/` for finished background runs.
2. **Ask Antoine for explicit confirmation before starting ANY compute-intensive run** (full eval, cache-cold VLM/SAM pass, overnight job). State what runs, expected duration, and that it survives session end. If declined → preparation work only. **Downloads are exempt** — the network is fast and downloads are cheap (Antoine, 2026-07-23); fetch models freely, no confirmation needed.
3. Heavy runs are detached: `nohup <cmd> > experiments/2026-07-model-review/logs/<track>-<date>.log 2>&1 &`
4. One model eval at a time (12-thread CPU — concurrent evals thrash).
5. All heavy inference must go through a per-image cache under `cache/` BEFORE any full run.
6. End of session: update this file + `memory/model-evaluations.md`, commit.

CPU budget: always-on ≤ ~5s/image; fallback-tier (VLM-style, ~24/122 images, cached) ≤ ~60s/image; >120s/image → not viable in any role.

## Results table

| Date | Track | Config | IoU | Latency/img | Log | Decision |
|------|-------|--------|-----|-------------|-----|----------|
| 2026-03-15 | baseline | 2B Q8_0, 512px, prompt V4 | 116/122 (95.1%) | ~20s (VLM images) | — | current default |

## Track A — Qwen3-VL-4B A/B

- [x] Verify VLM cache key separates models — **found & fixed 2026-07-23**: `vlm_model` stayed at the 2B ID regardless of `--vlm-gguf`, so 4B would have silently reused the 2B cache. Cache key now appends the GGUF stem for non-default models (`detector.py::_run_qwen_vlm`); existing 2B cache entries remain valid.
- [x] `download_models.py --vlm --vlm-size 4b` support added (Q8_0 ~4.3GB + mmproj F16 ~836MB)
- [x] Download 4B GGUFs (~5.1GB): **done 2026-07-23**, both files verified in `models/qwen3vl/` (Q8_0 4280MB + mmproj 836MB, `logs/trackA-4b-download-2026-07-23.log`)
- [ ] Run eval (**needs confirmation, est. 30-60 min cache-cold**, survives session end):
  ```
  cd ~/stuff/picture-frame-preprocessor && nohup venv/bin/python scripts/evaluate_art_class_accuracy.py --vlm \
    --vlm-gguf models/qwen3vl/Qwen3VL-4B-Instruct-Q8_0.gguf \
    --vlm-mmproj models/qwen3vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf \
    > experiments/2026-07-model-review/logs/trackA-4b-$(date +%F).log 2>&1 &
  ```
- [ ] Compare vs baseline: overall + per-class + the 6 misses (20200525, 20210910_203723, DSC_4162, DSC_4291, 2 text-filtered). Record per-image VLM latency from log.
- [ ] Decision: adopt only if net-positive IoU AND ≤ ~60s/image. Document either way in `memory/model-evaluations.md`.
- Guardrails: keep 512px, prompt V4, trigger heuristics unchanged. 8B = optional data point only (~80-100s/img, not a default candidate).

## Track B — SAM 3 prototype

Roles on the table: fallback-tier (like VLM) or crop-refinement on primary bbox ONLY. Ensemble replacement is off the table on this CPU regardless of accuracy.

- [ ] Write `scripts/evaluate_sam3.py` modeled on `scripts/evaluate_qwen3vl.py`, with versioned per-image cache (`cache/sam3/`), bf16
- [ ] Latency check on ~5 images first (**needs confirmation even for 5** — cache-cold SAM 3 is heavy). Budget: ≤60s → fallback-tier viable; 60-120s → crop-refinement only; >120s → try EfficientSAM3 or abandon
- [ ] Eval on 17-miss set (vs Qwen3-VL-2B's 14/17) + mosaic GT F1 (vs CLIP-region 0.529, OWLv2 0.293). Explicitly check the OWLv2 failure mode (fires on colorful murals?)
- [ ] Crop-tightening prototype on the 5 "excessive sky/wall" images from the March visual review
- [ ] Integration decision (opt-in flag only, `defaults.py`, full `/eval` + `/report` before any default change)

## Track C — LocateAnything-3B (conditional — only if A/B leave VLM-addressable misses open)

- [ ] Build llama.cpp `mtmd-grounders` fork branch in `~/stuff/llama.cpp-grounders/` (do NOT touch `~/stuff/llama.cpp/build/`); use `LLAMA_SERVER_BIN` to point at it; llama-server needs `--special` for grounding tokens
- [ ] Download `yuuko-eth/LocateAnything-3B-GGUF` Q4_K_M (~2.1GB + projector) — downloads exempt from confirmation
- [ ] Standalone eval on 17-miss set, same protocol as `evaluate_qwen3vl.py`
- [ ] Integrate only if it clearly beats the best Qwen result

## Track D — Processing speed

- [ ] D1: Grounding DINO → OpenVINO (biggest expected win; DINO tiny is the slowest ensemble stage). Follow upstream `demo/export_openvino.py` / OpenVINO blog recipe. Acceptance: same 116/122 **hit set** + measured speedup. Wire into existing `use_openvino` path in `detector.py`
- [ ] D2: YOLO-World OpenVINO re-check (`export_to_openvino.py:92` skips world models — retry with current ultralytics). Same hit-set-parity acceptance
- [ ] D3: llama.cpp rebuild (`~/stuff/llama.cpp/`, pull master + rebuild) — benchmark VLM s/image before/after on ~5 images with cache bypassed
- [ ] Benchmark methodology: time full 122-image eval from tee'd logs, report per-stage means

## Session log

- **2026-07-23 (session 1, prep only)**: Researched new options since March. Chose 4 tracks. Fixed VLM cache-key collision bug (would have invalidated the 4B A/B). Added `--vlm-size` to download script. Created this scaffold. No heavy compute run. Next: confirm 4B download + eval launch (Track A), D3 rebuild can interleave.
