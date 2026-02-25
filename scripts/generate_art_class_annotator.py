#!/usr/bin/env python3
"""Generate reports/art_class_annotator.html — art class annotator with bbox drawing.

Features:
  - Draw bounding boxes on each image (click+drag)
  - Assign an art class to each box
  - Multiple boxes with different classes on the same image
  - Delete individual boxes
  - Pre-populated from art_class_ground_truth.json (whole-image labels converted to full-image boxes)
  - Export to art_class_ground_truth.json format

Usage:
    venv/bin/python scripts/generate_art_class_annotator.py
"""

import json
from pathlib import Path

GT_PATH = Path("test_real_images/ground_truth_annotations.json")
ART_CLASS_GT_PATH = Path("test_real_images/art_class_ground_truth.json")
OUTPUT_PATH = Path("reports/art_class_annotator.html")

# Canonical art classes: (id, label, keyboard shortcut, hex color)
ART_CLASSES = [
    ("painting",     "Painting",      "1", "#4a90d9"),
    ("mural",        "Mural",         "2", "#9b59b6"),
    ("mosaic",       "Mosaic",        "3", "#e67e22"),
    ("sculpture",    "Sculpture",     "4", "#27ae60"),
    ("ceramic",      "Ceramic/Vase",  "5", "#e91e63"),
    ("street_art",   "Street Art",    "6", "#e74c3c"),
    ("installation", "Installation",  "7", "#16a085"),
    ("non_art",      "Non-Art",       "8", "#7f8c8d"),
]


def main():
    with open(GT_PATH) as f:
        gt_list = json.load(f)
    gt_by_fn = {e["filename"]: e for e in gt_list}

    with open(ART_CLASS_GT_PATH) as f:
        existing = json.load(f)

    # Build IMAGES array
    images = []
    for fn in sorted(existing.keys()):
        entry = gt_by_fn.get(fn, {})
        orig_w = entry.get("original_width", 0)
        orig_h = entry.get("original_height", 0)
        pre_class = existing[fn].get("primary_class", "unknown")
        pre_conf  = existing[fn].get("confidence", "low")
        pre_notes = existing[fn].get("notes", "")

        # Pre-populate boxes: if a confident class was inferred, create a full-image box
        pre_boxes = []
        if pre_class not in ("unknown", "non_art") and pre_conf in ("high", "medium"):
            pre_boxes = [{"x1n": 0.0, "y1n": 0.0, "x2n": 1.0, "y2n": 1.0, "art_class": pre_class}]

        # Also include GT manual boxes from ground_truth_annotations (displayed as gold overlays)
        gt_boxes = []
        for b in entry.get("manual_boxes", []):
            bbox = b.get("bbox", [])
            if len(bbox) == 4 and orig_w and orig_h:
                x1, y1, x2, y2 = bbox
                gt_boxes.append({
                    "x1n": round(x1 / orig_w, 4),
                    "y1n": round(y1 / orig_h, 4),
                    "x2n": round(x2 / orig_w, 4),
                    "y2n": round(y2 / orig_h, 4),
                })

        images.append({
            "filename": fn,
            "pre_boxes": pre_boxes,
            "pre_notes": pre_notes,
            "gt_boxes": gt_boxes,   # gold overlay only, not editable
        })

    images_js   = json.dumps(images, indent=2)
    classes_js  = json.dumps([
        {"id": c[0], "label": c[1], "key": c[2], "color": c[3]}
        for c in ART_CLASSES
    ], indent=2)
    unknown_color = "#555555"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Art Class Annotator</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #1a1a1a; color: #e0e0e0;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}}
#header {{
  background: #2a2a2a; border-bottom: 2px solid #4a90d9;
  padding: 8px 16px; display: flex; align-items: center;
  gap: 12px; flex-shrink: 0; flex-wrap: wrap;
}}
#header h1 {{ color: #4a90d9; font-size: 1.1rem; }}
#progress-text {{ color: #FFD700; font-size: 0.9rem; font-weight: bold; }}
.btn {{
  padding: 5px 12px; border: none; border-radius: 4px; cursor: pointer;
  font-size: 0.82rem; font-weight: bold; background: #3a3a3a; color: #e0e0e0;
}}
.btn:hover {{ background: #555; }}
#main {{ display: flex; flex: 1; overflow: hidden; }}

/* ── Sidebar ── */
#sidebar {{
  width: 170px; flex-shrink: 0; background: #252525;
  border-right: 1px solid #333; overflow-y: auto; padding: 4px 0;
}}
.sidebar-item {{
  padding: 3px 8px; cursor: pointer; font-size: 0.72rem;
  border-left: 3px solid transparent;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.sidebar-item:hover {{ background: #333; }}
.sidebar-item.active {{ background: #333; border-left-color: #4a90d9; }}
.sidebar-item.done {{ color: #aaa; }}
.sidebar-item.empty {{ color: #FFD700; }}

/* ── Canvas ── */
#canvas-area {{
  flex: 1; display: flex; align-items: center; justify-content: center;
  background: #111; position: relative; overflow: hidden;
}}
#main-canvas {{ cursor: crosshair; display: block; }}

/* ── Right panel ── */
#right-panel {{
  width: 210px; flex-shrink: 0; background: #252525;
  border-left: 1px solid #333; display: flex; flex-direction: column;
  overflow: hidden;
}}
#class-section {{ padding: 10px 10px 6px; flex-shrink: 0; }}
#class-section h3 {{ color: #888; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 6px; }}
.class-btn {{
  width: 100%; padding: 6px 8px; border: 2px solid transparent;
  border-radius: 5px; cursor: pointer; font-size: 0.8rem; font-weight: bold;
  text-align: left; background: #333; color: #e0e0e0;
  display: flex; align-items: center; gap: 6px; margin-bottom: 3px;
}}
.class-btn:hover {{ filter: brightness(1.25); }}
.class-btn.active {{ border-color: white !important; }}
.key-badge {{
  background: rgba(0,0,0,0.4); border-radius: 3px; padding: 0 4px;
  font-size: 0.72rem; color: #ccc; font-family: monospace; flex-shrink: 0;
}}
#box-section {{
  flex: 1; overflow-y: auto; padding: 6px 10px;
  border-top: 1px solid #333;
}}
#box-section h3 {{ color: #888; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 6px; }}
.box-item {{
  display: flex; align-items: center; gap: 6px;
  padding: 4px 6px; border-radius: 4px; margin-bottom: 3px;
  font-size: 0.78rem; cursor: pointer;
}}
.box-item:hover {{ background: #333; }}
.box-item.selected {{ background: #3a3a3a; outline: 1px solid #666; }}
.box-color-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.box-delete {{ margin-left: auto; color: #e74c3c; cursor: pointer; font-size: 0.9rem; flex-shrink: 0; }}
.box-delete:hover {{ color: #ff6b6b; }}
#notes-section {{ padding: 6px 10px 10px; border-top: 1px solid #333; flex-shrink: 0; }}
#notes-section h3 {{ color: #888; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px; }}
#notes-input {{
  width: 100%; background: #333; border: 1px solid #555; color: #ddd;
  border-radius: 4px; padding: 5px; font-size: 0.78rem; resize: vertical; min-height: 50px;
}}
#filename-label {{ padding: 6px 10px 0; font-size: 0.7rem; color: #777; word-break: break-all; }}
</style>
</head>
<body>
<div id="header">
  <h1>Art Class Annotator</h1>
  <span id="progress-text">...</span>
  <button class="btn" style="font-size:1rem;padding:3px 10px" onclick="navigate(-1)" title="Previous (←)">&#8592;</button>
  <button class="btn" style="font-size:1rem;padding:3px 10px" onclick="navigate(1)" title="Next (→ or Space)">&#8594; <span style="font-size:0.7rem;opacity:0.7">Space</span></button>
  <button class="btn" onclick="jumpToEmpty()">Next Empty</button>
  <button class="btn" onclick="clearImage()" style="color:#e74c3c">Clear Image</button>
  <button class="btn" onclick="exportJSON()" style="background:#2d6a4f">Export JSON</button>
  <span style="font-size:0.78rem;color:#888">Draw to add box (assigned to active class). Click to select, Del to delete.</span>
  <span style="font-size:0.78rem;color:#FFD700;margin-left:6px">- - - Gold dashed = previous GT crop reference (read-only)</span>
</div>
<div id="main">
  <div id="sidebar"></div>
  <div id="canvas-area">
    <canvas id="main-canvas"></canvas>
  </div>
  <div id="right-panel">
    <div id="filename-label">—</div>
    <div id="class-section">
      <h3>Active Class (draw next box as)</h3>
      <div id="class-buttons"></div>
    </div>
    <div id="box-section">
      <h3>Boxes on this image</h3>
      <div id="box-list"></div>
    </div>
    <div id="notes-section">
      <h3>Notes</h3>
      <textarea id="notes-input" placeholder="Optional notes..."></textarea>
    </div>
  </div>
</div>
<script>
// ─────────────────────────────────────────────
// DATA
// ─────────────────────────────────────────────
const IMAGES = {images_js};
const ART_CLASSES = {classes_js};
const UNKNOWN_COLOR = '{unknown_color}';
const STORAGE_KEY = 'art_class_annotations_v2';

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
let currentIdx = 0;
let annotations = {{}};   // filename → {{ boxes: [], notes: '' }}
let activeClass = ART_CLASSES[0].id;
let selectedBoxIdx = -1;  // index in current image's box list

// Drawing state
let drawing = false;
let dragStart = null;   // {{cx, cy}} in canvas coords
let dragCurrent = null;

// Loaded image element (for drawing)
let loadedImg = null;
let canvasScale = 1;  // factor to map canvas px → display px (always 1 for us)

// ─────────────────────────────────────────────
// PERSISTENCE
// ─────────────────────────────────────────────
function loadFromStorage() {{
  try {{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) annotations = JSON.parse(raw);
  }} catch(e) {{ annotations = {{}}; }}
}}

function saveToStorage() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations)); }}

function getAnnotation(filename) {{
  if (!annotations[filename]) {{
    const img = IMAGES.find(i => i.filename === filename);
    annotations[filename] = {{
      boxes: img ? JSON.parse(JSON.stringify(img.pre_boxes)) : [],
      notes: img ? (img.pre_notes || '') : '',
    }};
  }}
  return annotations[filename];
}}

// ─────────────────────────────────────────────
// CANVAS & RENDER
// ─────────────────────────────────────────────
const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

function classColor(classId) {{
  const c = ART_CLASSES.find(c => c.id === classId);
  return c ? c.color : UNKNOWN_COLOR;
}}

function redraw() {{
  if (!loadedImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(loadedImg, 0, 0, canvas.width, canvas.height);

  const img_data = IMAGES[currentIdx];
  const ann = getAnnotation(img_data.filename);
  const W = canvas.width, H = canvas.height;

  // Draw GT boxes (gold, dashed)
  ctx.save();
  ctx.strokeStyle = '#FFD700';
  ctx.lineWidth = Math.max(2, W / 400);
  ctx.setLineDash([8, 4]);
  for (const b of (img_data.gt_boxes || [])) {{
    ctx.strokeRect(b.x1n * W, b.y1n * H, (b.x2n - b.x1n) * W, (b.y2n - b.y1n) * H);
  }}
  ctx.setLineDash([]);
  ctx.restore();

  // Draw annotated boxes
  for (let i = 0; i < ann.boxes.length; i++) {{
    const b = ann.boxes[i];
    const col = classColor(b.art_class);
    const lw = Math.max(2, W / 350);
    ctx.strokeStyle = col;
    ctx.lineWidth = i === selectedBoxIdx ? lw * 2 : lw;
    ctx.strokeRect(b.x1n * W, b.y1n * H, (b.x2n - b.x1n) * W, (b.y2n - b.y1n) * H);
    // Class label
    const label = (ART_CLASSES.find(c => c.id === b.art_class) || {{label: b.art_class}}).label;
    const fontSize = Math.max(11, Math.min(16, W / 60));
    ctx.font = `bold ${{fontSize}}px sans-serif`;
    const tx = b.x1n * W + 4;
    const ty = b.y1n * H + fontSize + 2;
    ctx.fillStyle = col + 'cc';
    const tw = ctx.measureText(label).width;
    ctx.fillRect(tx - 2, ty - fontSize, tw + 6, fontSize + 4);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, tx + 1, ty);
  }}

  // Draw in-progress box
  if (drawing && dragStart && dragCurrent) {{
    const col = classColor(activeClass);
    ctx.strokeStyle = col;
    ctx.lineWidth = Math.max(2, W / 400);
    ctx.setLineDash([6, 3]);
    const x = Math.min(dragStart.x, dragCurrent.x);
    const y = Math.min(dragStart.y, dragCurrent.y);
    const w = Math.abs(dragCurrent.x - dragStart.x);
    const h = Math.abs(dragCurrent.y - dragStart.y);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }}
}}

function loadImageAndRender() {{
  const img_data = IMAGES[currentIdx];
  const ann = getAnnotation(img_data.filename);

  selectedBoxIdx = -1;
  document.getElementById('filename-label').textContent = img_data.filename;
  document.getElementById('notes-input').value = ann.notes || '';

  const imgEl = new Image();
  imgEl.onload = () => {{
    // Scale canvas to fit in the available area
    const area = document.getElementById('canvas-area');
    const maxW = area.clientWidth - 4;
    const maxH = area.clientHeight - 4;
    const scale = Math.min(1, maxW / imgEl.naturalWidth, maxH / imgEl.naturalHeight);
    canvas.width  = Math.round(imgEl.naturalWidth  * scale);
    canvas.height = Math.round(imgEl.naturalHeight * scale);
    loadedImg = imgEl;
    redraw();
  }};
  imgEl.src = '../test_real_images/input/' + img_data.filename;

  updateBoxList();
  updateSidebar();
  updateProgress();
  updateClassButtons();
}}

function updateBoxList() {{
  const img_data = IMAGES[currentIdx];
  const ann = getAnnotation(img_data.filename);
  const list = document.getElementById('box-list');
  list.innerHTML = '';
  if (ann.boxes.length === 0) {{
    list.innerHTML = '<div style="color:#666;font-size:0.78rem">No class boxes yet.<br>Draw on the image to label subjects.<br><span style="color:#FFD700">Gold dashed = GT crop reference.</span></div>';
    return;
  }}
  ann.boxes.forEach((b, i) => {{
    const col = classColor(b.art_class);
    const label = (ART_CLASSES.find(c => c.id === b.art_class) || {{label: b.art_class}}).label;
    const div = document.createElement('div');
    div.className = 'box-item' + (i === selectedBoxIdx ? ' selected' : '');
    div.innerHTML = `<span class="box-color-dot" style="background:${{col}}"></span>
      <span>${{i + 1}}. ${{label}}</span>
      <span class="box-delete" title="Delete box" data-idx="${{i}}">&#x2715;</span>`;
    div.onclick = (e) => {{
      if (e.target.classList.contains('box-delete')) {{
        deleteBox(parseInt(e.target.dataset.idx));
      }} else {{
        selectedBoxIdx = i;
        updateBoxList();
        redraw();
      }}
    }};
    list.appendChild(div);
  }});
}}

function updateClassButtons() {{
  document.querySelectorAll('.class-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.dataset.classId === activeClass);
  }});
}}

function updateSidebar() {{
  const sidebar = document.getElementById('sidebar');
  sidebar.innerHTML = '';
  IMAGES.forEach((img_data, idx) => {{
    const ann = annotations[img_data.filename];
    const hasPre = img_data.pre_boxes && img_data.pre_boxes.length > 0;
    const hasBoxes = ann && ann.boxes && ann.boxes.length > 0;
    const isEmpty = !hasBoxes && !hasPre;
    const base = img_data.filename.split('.').slice(0, -1).join('.');
    const div = document.createElement('div');
    div.className = 'sidebar-item' +
      (idx === currentIdx ? ' active' : '') +
      (isEmpty ? ' empty' : ' done');
    div.textContent = (idx + 1) + '. ' + base;
    div.title = img_data.filename;
    div.onclick = () => {{ currentIdx = idx; loadImageAndRender(); }};
    sidebar.appendChild(div);
  }});
  const active = sidebar.querySelector('.active');
  if (active) active.scrollIntoView({{ block: 'nearest' }});
}}

function updateProgress() {{
  const done = IMAGES.filter(img => {{
    const ann = annotations[img.filename];
    return (ann && ann.boxes && ann.boxes.length > 0) ||
           (img.pre_boxes && img.pre_boxes.length > 0);
  }}).length;
  document.getElementById('progress-text').textContent = done + ' / ' + IMAGES.length + ' annotated';
}}

// ─────────────────────────────────────────────
// NAVIGATION
// ─────────────────────────────────────────────
function navigate(dir) {{
  currentIdx = (currentIdx + dir + IMAGES.length) % IMAGES.length;
  loadImageAndRender();
}}

function jumpToEmpty() {{
  for (let i = 1; i <= IMAGES.length; i++) {{
    const idx = (currentIdx + i) % IMAGES.length;
    const img_data = IMAGES[idx];
    const ann = annotations[img_data.filename];
    const hasBoxes = (ann && ann.boxes && ann.boxes.length > 0) ||
                     (img_data.pre_boxes && img_data.pre_boxes.length > 0);
    if (!hasBoxes) {{
      currentIdx = idx;
      loadImageAndRender();
      return;
    }}
  }}
  alert('All images have at least one box!');
}}

// ─────────────────────────────────────────────
// ANNOTATION
// ─────────────────────────────────────────────
function setActiveClass(classId) {{
  activeClass = classId;
  updateClassButtons();
}}

function deleteBox(idx) {{
  const img_data = IMAGES[currentIdx];
  const ann = getAnnotation(img_data.filename);
  ann.boxes.splice(idx, 1);
  selectedBoxIdx = -1;
  saveToStorage();
  updateBoxList();
  updateSidebar();
  updateProgress();
  redraw();
}}

function clearImage() {{
  if (!confirm('Delete all boxes for this image?')) return;
  const img_data = IMAGES[currentIdx];
  const ann = getAnnotation(img_data.filename);
  ann.boxes = [];
  selectedBoxIdx = -1;
  saveToStorage();
  updateBoxList();
  updateSidebar();
  updateProgress();
  redraw();
}}

// ─────────────────────────────────────────────
// BOX DRAWING
// ─────────────────────────────────────────────
function canvasCoords(e) {{
  const rect = canvas.getBoundingClientRect();
  // Canvas logical px coords (canvas.width × canvas.height)
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {{
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  }};
}}

canvas.addEventListener('mousedown', e => {{
  if (e.button !== 0) return;
  drawing = true;
  dragStart = canvasCoords(e);
  dragCurrent = dragStart;
}});

canvas.addEventListener('mousemove', e => {{
  if (!drawing) return;
  dragCurrent = canvasCoords(e);
  redraw();
}});

canvas.addEventListener('mouseup', e => {{
  if (!drawing) return;
  drawing = false;
  const end = canvasCoords(e);

  const W = canvas.width, H = canvas.height;
  const x1n = Math.min(dragStart.x, end.x) / W;
  const y1n = Math.min(dragStart.y, end.y) / H;
  const x2n = Math.max(dragStart.x, end.x) / W;
  const y2n = Math.max(dragStart.y, end.y) / H;

  // Ignore tiny drags (< 1% width or height)
  if ((x2n - x1n) < 0.01 || (y2n - y1n) < 0.01) {{ redraw(); return; }}

  const img_data = IMAGES[currentIdx];
  const ann = getAnnotation(img_data.filename);
  ann.boxes.push({{
    x1n: Math.max(0, Math.round(x1n * 1000) / 1000),
    y1n: Math.max(0, Math.round(y1n * 1000) / 1000),
    x2n: Math.min(1, Math.round(x2n * 1000) / 1000),
    y2n: Math.min(1, Math.round(y2n * 1000) / 1000),
    art_class: activeClass,
  }});
  selectedBoxIdx = ann.boxes.length - 1;
  saveToStorage();
  updateBoxList();
  updateSidebar();
  updateProgress();
  redraw();
}});

canvas.addEventListener('mouseleave', () => {{
  if (drawing) {{ drawing = false; redraw(); }}
}});

// ─────────────────────────────────────────────
// KEYBOARD
// ─────────────────────────────────────────────
function setupKeys() {{
  document.addEventListener('keydown', e => {{
    if (document.activeElement === document.getElementById('notes-input')) return;
    switch(e.key) {{
      case 'ArrowLeft':  e.preventDefault(); navigate(-1); break;
      case 'ArrowRight':
      case ' ':          e.preventDefault(); navigate(1);  break;
      case 'Delete':
      case 'Backspace':
        if (selectedBoxIdx >= 0) {{ e.preventDefault(); deleteBox(selectedBoxIdx); }}
        break;
      default: {{
        const cls = ART_CLASSES.find(c => c.key === e.key);
        if (cls) {{ e.preventDefault(); setActiveClass(cls.id); }}
      }}
    }}
  }});
  document.getElementById('notes-input').addEventListener('blur', () => {{
    const img_data = IMAGES[currentIdx];
    const ann = getAnnotation(img_data.filename);
    ann.notes = document.getElementById('notes-input').value;
    saveToStorage();
  }});
}}

// ─────────────────────────────────────────────
// EXPORT
// ─────────────────────────────────────────────
function exportJSON() {{
  const output = {{}};
  IMAGES.forEach(img_data => {{
    const ann = getAnnotation(img_data.filename);
    const boxes = ann.boxes || [];
    // Primary class = class of first (or largest) box
    let primaryClass = 'unknown';
    if (boxes.length > 0) {{
      // Largest box by area
      let best = boxes[0], bestArea = 0;
      for (const b of boxes) {{
        const area = (b.x2n - b.x1n) * (b.y2n - b.y1n);
        if (area > bestArea) {{ bestArea = area; best = b; }}
      }}
      primaryClass = best.art_class;
    }}
    output[img_data.filename] = {{
      primary_class: primaryClass,
      confidence: boxes.length > 0 ? 'high' : 'unknown',
      boxes: boxes.map(b => ({{
        x1_norm: b.x1n, y1_norm: b.y1n, x2_norm: b.x2n, y2_norm: b.y2n,
        art_class: b.art_class,
      }})),
      notes: ann.notes || '',
    }};
  }});
  const json = JSON.stringify(output, null, 2);
  const blob = new Blob([json], {{ type: 'application/json' }});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'art_class_ground_truth.json';
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}}

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
function init() {{
  loadFromStorage();

  // Build class buttons
  const container = document.getElementById('class-buttons');
  ART_CLASSES.forEach(cls => {{
    const btn = document.createElement('button');
    btn.className = 'class-btn';
    btn.dataset.classId = cls.id;
    btn.style.background  = cls.color + '33';
    btn.style.borderColor = cls.color + '88';
    btn.innerHTML = `<span class="key-badge">${{cls.key}}</span> ${{cls.label}}`;
    btn.onclick = () => setActiveClass(cls.id);
    container.appendChild(btn);
  }});

  // Start at first empty image
  const firstEmpty = IMAGES.findIndex(img => {{
    const ann = annotations[img.filename];
    return (!ann || !ann.boxes || ann.boxes.length === 0) &&
           (!img.pre_boxes || img.pre_boxes.length === 0);
  }});
  currentIdx = firstEmpty >= 0 ? firstEmpty : 0;

  setupKeys();
  loadImageAndRender();
}}

init();
</script>
</body>
</html>"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    need_annotation = sum(
        1 for img in images
        if not img["pre_boxes"] and not img["gt_boxes"]
    )
    print(f"Written to {OUTPUT_PATH}")
    print(f"  {len(images)} images, {len([i for i in images if i['pre_boxes']])} pre-populated, {need_annotation} need manual boxes")
    print(f"Open: file://{OUTPUT_PATH.absolute()}")


if __name__ == "__main__":
    main()
