Run the detection evaluation pipeline and compare results against the stored baseline.

Steps:
1. Read the current baseline from `eval-baseline.json`
2. Run: `venv/bin/python scripts/evaluate_art_class_accuracy.py --vlm 2>&1 | tee /tmp/eval_out.txt`
3. Parse the output to extract IoU hit rate and per-class breakdown
4. Compare against the baseline and report a clear summary table showing:
   - Overall IoU: baseline vs current (with delta)
   - Per-class breakdown with deltas for any changes
   - List of specific images that changed status (new hits or new misses)
5. If results improved, ask whether to update `eval-baseline.json` with the new numbers

If the user passes `$ARGUMENTS`, append them to the eval command (e.g., `/eval --no-cache` or `/eval --verbose`).

Important: The eval takes ~2-5 minutes without VLM cache, ~30s with cache. Run it in background if the user has other work to do.
