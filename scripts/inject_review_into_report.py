#!/usr/bin/env python3
"""Inject visual review findings into the interactive report HTML.

Reads findings from /tmp/visual_review/findings.json and adds them
as annotations visible in the right panel of the report.

Usage:
    venv/bin/python scripts/inject_review_into_report.py [findings.json]
"""

import json
import sys
from pathlib import Path

REPORT = Path("reports/interactive_detection_report.html")
FINDINGS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/visual_review/findings.json")

if not FINDINGS.exists():
    print(f"Findings not found: {FINDINGS}")
    sys.exit(1)

with open(FINDINGS) as f:
    findings = json.load(f)

with open(REPORT) as f:
    html = f.read()

# Build the findings data as a JS object keyed by filename
findings_js = json.dumps(findings)

# CSS for the review section
review_css = """
#review-section {
  padding: 6px 10px; border-top: 1px solid #333; flex-shrink: 0;
  max-height: 180px; overflow-y: auto;
}
#review-section h3 { margin: 0 0 4px; font-size: 0.72rem; color: #aaa; }
.review-item {
  font-size: 0.68rem; padding: 3px 5px; margin: 2px 0;
  border-radius: 3px; line-height: 1.3;
}
.review-item.CROP_VALIDITY { background: #3a2a00; color: #ffcc44; }
.review-item.MISSED_OPP { background: #1a2a3a; color: #66aaff; }
.review-ok { font-size: 0.68rem; color: #4a4; padding: 3px 5px; }
"""

# JS to render review comments when an image is selected
review_js = """
const REVIEW_FINDINGS = __FINDINGS__;

function renderReview() {
  const el = document.getElementById('review-section');
  if (!el) return;
  const r = RESULTS[currentIdx];
  const items = REVIEW_FINDINGS[r.filename] || [];
  if (items.length === 0) {
    el.innerHTML = '<div class="review-ok">No review issues</div>';
    return;
  }
  el.innerHTML = '<h3>Review</h3>' + items.map(item =>
    `<div class="review-item ${item.type}">[${item.type}] ${item.desc}</div>`
  ).join('');
}
""".replace("__FINDINGS__", findings_js)

# Inject CSS before </style>
html = html.replace("</style>", review_css + "\n</style>", 1)

# Inject the review section div after badges-section
html = html.replace(
    '<div id="info-section"></div>',
    '<div id="review-section"></div>\n    <div id="info-section"></div>',
    1
)

# Inject review data+function right after RESULTS declaration so it's
# defined before renderContent() can call renderReview().
html = html.replace(
    "\nfunction buildConfigPanel",
    "\n" + review_js + "\nfunction buildConfigPanel",
    1
)

# Add renderReview() call inside renderContent
html = html.replace(
    "document.getElementById('filename-label').textContent = r.filename;",
    "document.getElementById('filename-label').textContent = r.filename;\n  renderReview();",
    1
)

# Show the review filter button and set its count
n_with_issues = len(findings)
html = html.replace(
    'id="review-filter-btn" style="display:none">Review Issues</button>',
    f'id="review-filter-btn">Review Issues ({n_with_issues})</button>',
    1
)

with open(REPORT, "w") as f:
    f.write(html)

n_files = len(findings)
n_issues = sum(len(v) for v in findings.values())
print(f"Injected {n_issues} review findings for {n_files} files into {REPORT}")
