Visually inspect crop results using Sonnet subagents that read separate scene and crop images.

Each image produces two files: `{name}_scene.jpg` (original with boxes/outlines) and `{name}_crops.jpg` (final crop panels only). Agents read BOTH per image but use them for different assessments.

Steps:
1. Generate review images:
   ```
   venv/bin/python scripts/generate_visual_review.py $ARGUMENTS 2>&1 | tee /tmp/visual_review_gen.txt
   ```

2. Read `/tmp/visual_review/review_metadata.json` for the image list.

3. Split into batches of ~15 and launch Sonnet subagents in parallel (`model: "sonnet"`, `run_in_background: true`). For each image, the agent reads TWO files:
   - First: `{name}_crops.jpg` — the final crop panels (what the picture frame displays)
   - Second: `{name}_scene.jpg` — the original scene with detection boxes and crop outlines

   For each image, evaluate:

   **A. Crop validity (judge from _crops.jpg ONLY):**
   - Does each crop show art content? Flag non-art (people, signs, QR codes, blank walls).
   - Are focal crops (zoomed details at 1.5x-4x within a primary at 1.0x) zooming into interesting art details? These are intentionally designed — only flag if target is blank/damaged/signage.
   - For duplicates: read overlap % at bottom of _crops.jpg. If < 70% → NOT duplicates, STOP.
   - Flag extremely blurry over-zooms.

   **B. Missed opportunities (judge from _scene.jpg ONLY):**
   - Are there art subjects visible but NOT covered by any crop outline (C1/C2/C3)?
   - Is the GT box (green) significantly different from crop outlines?
   - For single-crop images: are there additional distinct art subjects?

   Return JSON: `[{"file": "f.jpg", "type": "CROP_VALIDITY|MISSED_OPP", "desc": "..."}]`
   End with: `GOOD: N images looked correct`

4. Collect results:
   ```
   venv/bin/python scripts/collect_review_findings.py <output_files...>
   ```

5. Inject into report:
   ```
   venv/bin/python scripts/inject_review_into_report.py
   ```

6. Present unified summary.

---

**Prompt for agents:**

You review art photo crops for a picture frame display. For each image you will read TWO separate files:
1. `_crops.jpg` — ONLY the final crop panels. This is what gets displayed on the frame.
2. `_scene.jpg` — The original photo with detection boxes and crop region outlines.

**Hard constraint — fixed aspect ratio crop:**
The crops are a fixed 480x800 portrait window extracted from the source photo. The cropper cannot freely resize or reshape the window — it can only slide it horizontally/vertically and apply zoom. If the artwork is wider than the crop window, the crop MUST cut off the edges — this is unavoidable, not a bug. Don't flag edge clipping on wide artworks as a crop issue unless the crop is clearly mis-centered (e.g., showing blank wall on one side while cutting art on the other).

**Rules:**
- **Crop quality (framing, clipping, content):** Judge ONLY from `_crops.jpg`. Never describe what the subject looks like in the scene image — the cropper applies zoom/centering that changes the framing significantly.
- **Missed opportunities (uncropped art, GT mismatch):** Judge ONLY from `_scene.jpg`. Look at crop outlines to see what IS being captured.
- **Focal crop pattern (GOOD):** When C1 is at 1.0x (full artwork), additional crops at higher zoom on faces/figures/details are intentional variety for the picture frame. Only flag if zoom target is blank/damaged/signage, or two focal crops target the SAME detail.
- **Duplicates:** Read overlap % at bottom of `_crops.jpg`. If < 70% → NOT duplicates, STOP.
- Focus on ACTIONABLE issues only. Be concise: one line per issue.
- **ONLY return issues.** Do NOT include positive observations ("good crop", "correct", "well-framed") in the JSON array. If an image has no issues, simply don't include it. The `GOOD: N` count at the end is sufficient.
