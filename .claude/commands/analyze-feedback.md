Analyze an exported feedback JSON file from the interactive report.

The user should provide a path to the feedback JSON file as $ARGUMENTS. If not provided, look for the most recent file matching `reports/feedback/detection_feedback_*.json` or `test_real_images/detection_feedback_*.json`.

Steps:
1. Read the feedback JSON file
2. Summarize counts by category (good, bad_detection, bad_crop, bad_both, skipped/unreviewed)
3. For ALL images (including "good" ones), check for textual comments/notes — these contain valuable context even on images marked as good
4. For each "bad" category, list the affected image filenames
5. List all textual feedback verbatim, grouped by category
6. Compare with previous feedback files if available (look for older files in the same directory) to show improvement trends
7. Group bad_detection cases by likely root cause:
   - Wrong primary selection (correct subject exists but scored lower)
   - Subject not detected (no detection covers the ground truth)
   - Misclassification (wrong class label)
8. Suggest which issues are most impactful to fix based on frequency
