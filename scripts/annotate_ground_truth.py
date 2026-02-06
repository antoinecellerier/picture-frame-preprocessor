#!/usr/bin/env python3
"""Generate interactive HTML for annotating ground truth bounding boxes.

Uses the OptimizedEnsembleDetector (YOLO-World + Grounding DINO) to show
detected bounding boxes on each image. You can then select correct detections
or draw manual ground truth boxes, and export the result as a JSON file
compatible with ground_truth_annotations.json.

By default, only processes images not yet in ground_truth_annotations.json.
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageOps
import base64
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frame_prep.detector import OptimizedEnsembleDetector

GROUND_TRUTH_PATH = Path('test_real_images/ground_truth_annotations.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML for annotating ground truth bounding boxes.\n'
                    'By default, only processes images not yet in ground_truth_annotations.json.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  %(prog)s                          # Annotate only new images\n'
               '  %(prog)s --all                     # Annotate all images (ignore existing)\n'
               '  %(prog)s --reannotate              # Edit existing annotations\n'
               '  %(prog)s --input-dir other/images/  # Use a different image directory\n'
    )
    parser.add_argument('--all', action='store_true',
                        help='Process ALL images, including already-annotated ones')
    parser.add_argument('--reannotate', action='store_true',
                        help='Process ALL images and pre-load existing annotations for editing')
    parser.add_argument('--input-dir', '-i', metavar='DIR',
                        default='test_real_images/input/',
                        help='Input image directory (default: test_real_images/input/)')
    parser.add_argument('--output-dir', metavar='DIR',
                        default='reports/',
                        help='Output directory for HTML (default: reports/)')
    return parser.parse_args()


def load_existing_annotations():
    """Load existing ground truth annotations if available."""
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH, 'r') as f:
            return json.load(f)
    return []


def get_annotated_filenames(annotations):
    """Get set of filenames that already have annotations."""
    return {entry['filename'] for entry in annotations}


def get_existing_annotation(annotations, filename):
    """Get existing annotation for a filename, or None."""
    for entry in annotations:
        if entry['filename'] == filename:
            return entry
    return None


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Get all test images
    test_images = sorted([
        f for f in input_dir.glob('*')
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    print(f"Found {len(test_images)} images in {input_dir}")

    # Load existing annotations
    existing_annotations = load_existing_annotations()
    annotated_filenames = get_annotated_filenames(existing_annotations)
    print(f"Found {len(annotated_filenames)} existing annotations in {GROUND_TRUTH_PATH}")

    # Filter images based on flags
    if args.reannotate or args.all:
        images_to_process = test_images
        mode = "reannotate" if args.reannotate else "all"
        print(f"Mode: {'re-annotate (pre-loading existing boxes)' if args.reannotate else 'all images (fresh start)'}")
    else:
        images_to_process = [
            img for img in test_images
            if img.name not in annotated_filenames
        ]
        mode = "new_only"
        print(f"Mode: new images only")

    if not images_to_process:
        print("\nNo images to annotate! All images already have ground truth.")
        print("Use --all or --reannotate to re-process existing images.")
        sys.exit(0)

    print(f"\nProcessing {len(images_to_process)} images...")

    # Create detector once (reused for all images)
    print("Loading detector models...")
    detector = OptimizedEnsembleDetector(
        confidence_threshold=0.25,
        merge_threshold=0.2,
        two_pass=True
    )

    # Build pre-loaded annotations lookup for --reannotate mode
    preloaded = {}
    if args.reannotate:
        for entry in existing_annotations:
            preloaded[entry['filename']] = {
                'manual_boxes': entry.get('manual_boxes', []),
                'correct_detections': entry.get('correct_detections', []),
            }

    # Store all detection data
    all_data = []

    for idx, img_path in enumerate(images_to_process):
        filename = img_path.name
        print(f"  [{idx+1}/{len(images_to_process)}] Processing {filename}...")

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Resize for display (max 900px width)
        display_img = img.copy()
        scale = 900 / display_img.width if display_img.width > 900 else 1.0
        if scale < 1.0:
            new_size = (900, int(display_img.height * scale))
            display_img = display_img.resize(new_size, Image.Resampling.LANCZOS)

        orig_b64 = image_to_base64(display_img)

        # Run detection
        try:
            detections = detector.detect(img, verbose=False, image_path=img_path)
        except TypeError:
            detections = detector.detect(img, verbose=False)

        # Collect detections
        all_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            all_detections.append({
                'class_name': det.class_name,
                'confidence': float(det.confidence),
                'bbox': [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)],
            })

        # Pre-loaded annotations for --reannotate (scale to display coords)
        preloaded_manual = []
        preloaded_correct = []
        if filename in preloaded:
            for box in preloaded[filename].get('manual_boxes', []):
                bx = box['bbox']
                preloaded_manual.append({
                    'bbox': [int(bx[0] * scale), int(bx[1] * scale),
                             int(bx[2] * scale), int(bx[3] * scale)]
                })
            for det in preloaded[filename].get('correct_detections', []):
                bx = det['bbox']
                preloaded_correct.append({
                    'class_name': det.get('class_name', ''),
                    'confidence': det.get('confidence', 0),
                    'bbox': [int(bx[0] * scale), int(bx[1] * scale),
                             int(bx[2] * scale), int(bx[3] * scale)]
                })

        all_data.append({
            'idx': idx,
            'filename': filename,
            'image_b64': orig_b64,
            'width': display_img.width,
            'height': display_img.height,
            'original_width': img.width,
            'original_height': img.height,
            'scale': scale,
            'detections': all_detections,
            'preloaded_manual': preloaded_manual,
            'preloaded_correct': preloaded_correct,
        })

    # Serialize existing annotations for merge on export
    existing_annotations_json = json.dumps(existing_annotations)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ground Truth Annotation Tool</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }}
        h1 {{
            text-align: center;
            color: #4CAF50;
            margin-bottom: 10px;
        }}
        .controls {{
            position: sticky;
            top: 0;
            background: #2a2a2a;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            border: 2px solid #4CAF50;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        }}
        .controls button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }}
        .controls button:hover {{
            background: #45a049;
        }}
        .summary {{
            display: inline-block;
            margin-left: 20px;
            font-size: 18px;
            color: #FFD700;
        }}
        .legend {{
            background: #333;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .legend h3 {{
            margin-top: 0;
            color: #64B5F6;
        }}
        .color-key {{
            display: flex;
            gap: 20px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        .color-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .color-box {{
            width: 30px;
            height: 20px;
            border: 2px solid white;
        }}
        .image-container {{
            background: #2a2a2a;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            border: 2px solid #444;
        }}
        .image-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .filename {{
            font-size: 20px;
            font-weight: bold;
            color: #FFD700;
        }}
        .image-controls {{
            display: flex;
            gap: 10px;
        }}
        .image-controls button {{
            background: #555;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .image-controls button:hover {{
            background: #666;
        }}
        .image-controls button.active {{
            background: #4CAF50;
        }}
        .image-controls button.draw-mode {{
            background: #FF6B6B;
        }}
        .image-controls button.draw-mode.active {{
            background: #FF3333;
        }}
        .canvas-wrapper {{
            position: relative;
            display: inline-block;
            border: 2px solid #555;
        }}
        canvas {{
            display: block;
        }}
        canvas.draw-mode {{
            cursor: crosshair;
        }}
        canvas.select-mode {{
            cursor: pointer;
        }}
        .detection-list {{
            margin-top: 15px;
            max-height: 200px;
            overflow-y: auto;
            background: #333;
            padding: 10px;
            border-radius: 4px;
        }}
        .detection-item {{
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid transparent;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .detection-item:hover {{
            background: #444;
        }}
        .detection-item.selected {{
            border-color: #FFD700;
            background: #3a3a2a;
        }}
        .detection-item.manual {{
            background: #3a2a3a;
            border-color: #FF00FF;
        }}
        .detection-item.manual.selected {{
            border-color: #FFD700;
            background: #4a2a4a;
        }}
        .detection-info {{
            flex: 1;
        }}
        .detection-model {{
            font-weight: bold;
            margin-right: 10px;
        }}
        .delete-btn {{
            background: #c62828;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }}
        .delete-btn:hover {{
            background: #b71c1c;
        }}
        #results {{
            background: #2a2a2a;
            padding: 20px;
            margin-top: 20px;
            border-radius: 8px;
            border: 2px solid #4CAF50;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
            display: none;
        }}
    </style>
</head>
<body>
    <h1>Ground Truth Annotation Tool ({len(images_to_process)} images)</h1>

    <div class="controls">
        <button onclick="exportGroundTruth()">Export Ground Truth JSON</button>
        <button onclick="generateResults()">Generate Summary</button>
        <button onclick="saveProgress()">Save Progress</button>
        <button onclick="loadProgress()">Load Progress</button>
        <span class="summary" id="progress-count">0/{len(images_to_process)} annotated</span>
    </div>

    <div class="legend">
        <h3>How to Use</h3>
        <p><strong>Mode 1: Select Existing Detections</strong></p>
        <p>Click on colored boxes from ensemble detections to mark them as correct ground truth.</p>
        <p>Multiple boxes can be selected if the subject spans multiple detections.</p>

        <p><strong>Mode 2: Draw Manual Ground Truth</strong></p>
        <p>Click "Draw Box" to enter drawing mode, then click and drag on the image.</p>
        <p>Manual boxes are shown in MAGENTA. Click "Draw Box" again to exit drawing mode.</p>

        <p><strong>Exporting:</strong></p>
        <p>Click "Export Ground Truth JSON" to download annotations.</p>
        <p>{'Exported file will be MERGED with existing annotations (keyed by filename).' if mode != 'all' else 'Exported file will contain only the images shown here.'}</p>
        <p>Copy the downloaded file to <code>test_real_images/ground_truth_annotations.json</code>.</p>

        <div class="color-key">
            <div class="color-item">
                <div class="color-box" style="background: #00BFFF;"></div>
                <span>Ensemble Detection</span>
            </div>
            <div class="color-item">
                <div class="color-box" style="background: #FF00FF;"></div>
                <span>Manual Ground Truth</span>
            </div>
            <div class="color-item">
                <div class="color-box" style="background: #FFD700;"></div>
                <span>Selected (gold border)</span>
            </div>
        </div>
    </div>

    <div id="images-container"></div>

    <div id="results"></div>

    <script>
        const imageData = {json.dumps(all_data)};
        const existingAnnotations = {existing_annotations_json};
        const detectionColor = '#00BFFF';
        const selections = {{}};
        const manualBoxes = {{}};
        const canvases = {{}};
        const contexts = {{}};
        const drawModes = {{}};
        const drawStart = {{}};

        function initImages() {{
            const container = document.getElementById('images-container');

            imageData.forEach(data => {{
                const div = document.createElement('div');
                div.className = 'image-container';
                div.id = `image-${{data.idx}}`;

                div.innerHTML = `
                    <div class="image-header">
                        <div class="filename">${{data.idx + 1}}. ${{data.filename}}</div>
                        <div class="image-controls">
                            <button onclick="toggleDrawMode(${{data.idx}})" class="draw-mode" id="draw-btn-${{data.idx}}">Draw Box</button>
                            <button onclick="clearImage(${{data.idx}})">Clear</button>
                        </div>
                    </div>
                    <div class="canvas-wrapper">
                        <canvas id="canvas-${{data.idx}}" width="${{data.width}}" height="${{data.height}}"></canvas>
                    </div>
                    <div class="detection-list" id="detections-${{data.idx}}"></div>
                `;

                container.appendChild(div);

                const canvas = document.getElementById(`canvas-${{data.idx}}`);
                const ctx = canvas.getContext('2d');
                canvases[data.idx] = canvas;
                contexts[data.idx] = ctx;
                selections[data.idx] = [];
                manualBoxes[data.idx] = [];
                drawModes[data.idx] = false;

                // Pre-load manual boxes from --reannotate
                if (data.preloaded_manual && data.preloaded_manual.length > 0) {{
                    data.preloaded_manual.forEach(box => {{
                        manualBoxes[data.idx].push({{ bbox: box.bbox }});
                    }});
                }}

                // Pre-select detections that match preloaded correct detections
                if (data.preloaded_correct && data.preloaded_correct.length > 0) {{
                    data.preloaded_correct.forEach(pc => {{
                        // Find the best matching detection by IoU
                        let bestIdx = -1;
                        let bestIoU = 0;
                        data.detections.forEach((det, detIdx) => {{
                            const iou = calcIoU(det.bbox, pc.bbox);
                            if (iou > bestIoU) {{
                                bestIoU = iou;
                                bestIdx = detIdx;
                            }}
                        }});
                        if (bestIdx >= 0 && bestIoU > 0.3 && !selections[data.idx].includes(bestIdx)) {{
                            selections[data.idx].push(bestIdx);
                        }}
                    }});
                }}

                const img = new Image();
                img.onload = () => {{
                    ctx.drawImage(img, 0, 0);
                    drawDetections(data.idx);
                }};
                img.src = 'data:image/jpeg;base64,' + data.image_b64;

                canvas.addEventListener('mousedown', (e) => handleMouseDown(e, data.idx));
                canvas.addEventListener('mousemove', (e) => handleMouseMove(e, data.idx));
                canvas.addEventListener('mouseup', (e) => handleMouseUp(e, data.idx));
                canvas.addEventListener('click', (e) => handleCanvasClick(e, data.idx));

                updateDetectionList(data.idx);
            }});

            updateProgress();
        }}

        function calcIoU(box1, box2) {{
            const x1 = Math.max(box1[0], box2[0]);
            const y1 = Math.max(box1[1], box2[1]);
            const x2 = Math.min(box1[2], box2[2]);
            const y2 = Math.min(box1[3], box2[3]);
            if (x2 <= x1 || y2 <= y1) return 0;
            const intersection = (x2 - x1) * (y2 - y1);
            const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
            const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
            return intersection / (area1 + area2 - intersection);
        }}

        function toggleDrawMode(imageIdx) {{
            drawModes[imageIdx] = !drawModes[imageIdx];
            const btn = document.getElementById(`draw-btn-${{imageIdx}}`);
            const canvas = canvases[imageIdx];

            if (drawModes[imageIdx]) {{
                btn.classList.add('active');
                canvas.className = 'draw-mode';
            }} else {{
                btn.classList.remove('active');
                canvas.className = 'select-mode';
            }}
        }}

        function getCanvasCoords(e, imageIdx) {{
            const canvas = canvases[imageIdx];
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {{
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            }};
        }}

        let isDrawing = false;

        function handleMouseDown(e, imageIdx) {{
            if (!drawModes[imageIdx]) return;
            isDrawing = true;
            const coords = getCanvasCoords(e, imageIdx);
            drawStart[imageIdx] = coords;
        }}

        function handleMouseMove(e, imageIdx) {{
            if (!drawModes[imageIdx] || !isDrawing || !drawStart[imageIdx]) return;

            const coords = getCanvasCoords(e, imageIdx);
            const start = drawStart[imageIdx];

            drawDetections(imageIdx);

            const ctx = contexts[imageIdx];
            ctx.strokeStyle = '#FF00FF';
            ctx.lineWidth = 4;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(
                start.x,
                start.y,
                coords.x - start.x,
                coords.y - start.y
            );
            ctx.setLineDash([]);
        }}

        function handleMouseUp(e, imageIdx) {{
            if (!drawModes[imageIdx] || !isDrawing || !drawStart[imageIdx]) return;
            isDrawing = false;

            const coords = getCanvasCoords(e, imageIdx);
            const start = drawStart[imageIdx];

            const width = Math.abs(coords.x - start.x);
            const height = Math.abs(coords.y - start.y);

            if (width > 10 && height > 10) {{
                const x1 = Math.min(start.x, coords.x);
                const y1 = Math.min(start.y, coords.y);
                const x2 = Math.max(start.x, coords.x);
                const y2 = Math.max(start.y, coords.y);

                manualBoxes[imageIdx].push({{
                    bbox: [x1, y1, x2, y2]
                }});

                drawDetections(imageIdx);
                updateDetectionList(imageIdx);
                updateProgress();
            }}

            drawStart[imageIdx] = null;
        }}

        function drawDetections(imageIdx) {{
            const data = imageData[imageIdx];
            const ctx = contexts[imageIdx];
            const img = new Image();

            img.onload = () => {{
                ctx.clearRect(0, 0, data.width, data.height);
                ctx.drawImage(img, 0, 0);

                // Draw ensemble detections
                data.detections.forEach((det, detIdx) => {{
                    const isSelected = selections[imageIdx].includes(detIdx);
                    const [x1, y1, x2, y2] = det.bbox;

                    ctx.strokeStyle = isSelected ? '#FFD700' : detectionColor;
                    ctx.lineWidth = isSelected ? 6 : 3;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    const label = `${{det.class_name}} ${{det.confidence.toFixed(2)}}`;
                    ctx.font = 'bold 14px Arial';
                    const textWidth = ctx.measureText(label).width;

                    ctx.fillStyle = isSelected ? '#FFD700' : detectionColor;
                    ctx.fillRect(x1, y1 - 22, textWidth + 8, 20);
                    ctx.fillStyle = 'black';
                    ctx.fillText(label, x1 + 4, y1 - 6);
                }});

                // Draw manual boxes
                manualBoxes[imageIdx].forEach((box, boxIdx) => {{
                    const [x1, y1, x2, y2] = box.bbox;

                    ctx.strokeStyle = '#FF00FF';
                    ctx.lineWidth = 5;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    ctx.fillStyle = '#FF00FF';
                    ctx.fillRect(x1, y1 - 22, 120, 20);
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px Arial';
                    ctx.fillText('GROUND TRUTH', x1 + 4, y1 - 6);
                }});
            }};
            img.src = 'data:image/jpeg;base64,' + data.image_b64;
        }}

        function handleCanvasClick(e, imageIdx) {{
            if (drawModes[imageIdx]) return;

            const coords = getCanvasCoords(e, imageIdx);
            const data = imageData[imageIdx];

            for (let i = data.detections.length - 1; i >= 0; i--) {{
                const det = data.detections[i];
                const [x1, y1, x2, y2] = det.bbox;
                if (coords.x >= x1 && coords.x <= x2 && coords.y >= y1 && coords.y <= y2) {{
                    const idx = selections[imageIdx].indexOf(i);
                    if (idx > -1) {{
                        selections[imageIdx].splice(idx, 1);
                    }} else {{
                        selections[imageIdx].push(i);
                    }}
                    drawDetections(imageIdx);
                    updateDetectionList(imageIdx);
                    updateProgress();
                    break;
                }}
            }}
        }}

        function deleteManualBox(imageIdx, boxIdx) {{
            manualBoxes[imageIdx].splice(boxIdx, 1);
            drawDetections(imageIdx);
            updateDetectionList(imageIdx);
            updateProgress();
        }}

        function updateDetectionList(imageIdx) {{
            const data = imageData[imageIdx];
            const listDiv = document.getElementById(`detections-${{imageIdx}}`);

            listDiv.innerHTML = '';

            // Manual boxes first
            manualBoxes[imageIdx].forEach((box, boxIdx) => {{
                const div = document.createElement('div');
                div.className = 'detection-item manual';
                div.innerHTML = `
                    <div class="detection-info">
                        <span class="detection-model" style="color: #FF00FF;">MANUAL</span>
                        <span>Ground Truth Box</span>
                    </div>
                    <button class="delete-btn" onclick="deleteManualBox(${{imageIdx}}, ${{boxIdx}})">Delete</button>
                `;
                listDiv.appendChild(div);
            }});

            // Ensemble detections
            data.detections.forEach((det, detIdx) => {{
                const isSelected = selections[imageIdx].includes(detIdx);
                const div = document.createElement('div');
                div.className = 'detection-item' + (isSelected ? ' selected' : '');
                div.onclick = () => {{
                    const idx = selections[imageIdx].indexOf(detIdx);
                    if (idx > -1) {{
                        selections[imageIdx].splice(idx, 1);
                    }} else {{
                        selections[imageIdx].push(detIdx);
                    }}
                    drawDetections(imageIdx);
                    updateDetectionList(imageIdx);
                    updateProgress();
                }};

                div.innerHTML = `
                    <div class="detection-info">
                        <span class="detection-model" style="color: ${{detectionColor}};">ENSEMBLE</span>
                        <span>${{det.class_name}} (${{det.confidence.toFixed(2)}})</span>
                    </div>
                    ${{isSelected ? '&#10003;' : ''}}
                `;
                listDiv.appendChild(div);
            }});
        }}

        function clearImage(imageIdx) {{
            selections[imageIdx] = [];
            manualBoxes[imageIdx] = [];
            drawDetections(imageIdx);
            updateDetectionList(imageIdx);
            updateProgress();
        }}

        function updateProgress() {{
            const annotated = imageData.filter(d =>
                selections[d.idx].length > 0 || manualBoxes[d.idx].length > 0
            ).length;
            document.getElementById('progress-count').textContent = `${{annotated}}/${{imageData.length}} annotated`;
        }}

        function exportGroundTruth() {{
            // Build annotations for the current images
            const newAnnotations = {{}};
            imageData.forEach(data => {{
                const result = {{
                    filename: data.filename,
                    original_width: data.original_width,
                    original_height: data.original_height,
                    scale: data.scale,
                    correct_detections: [],
                    manual_boxes: []
                }};

                // Add selected detections (scaled back to original size)
                selections[data.idx].forEach(detIdx => {{
                    const det = data.detections[detIdx];
                    const [x1, y1, x2, y2] = det.bbox;
                    result.correct_detections.push({{
                        model: 'ensemble',
                        class_name: det.class_name,
                        confidence: det.confidence,
                        bbox: [
                            Math.round(x1 / data.scale),
                            Math.round(y1 / data.scale),
                            Math.round(x2 / data.scale),
                            Math.round(y2 / data.scale)
                        ]
                    }});
                }});

                // Add manual boxes (scaled back to original size)
                manualBoxes[data.idx].forEach(box => {{
                    const [x1, y1, x2, y2] = box.bbox;
                    result.manual_boxes.push({{
                        bbox: [
                            Math.round(x1 / data.scale),
                            Math.round(y1 / data.scale),
                            Math.round(x2 / data.scale),
                            Math.round(y2 / data.scale)
                        ]
                    }});
                }});

                newAnnotations[data.filename] = result;
            }});

            // Merge with existing annotations (new annotations override by filename)
            const existingByFilename = {{}};
            existingAnnotations.forEach(entry => {{
                existingByFilename[entry.filename] = entry;
            }});

            // Override existing with new
            Object.keys(newAnnotations).forEach(filename => {{
                existingByFilename[filename] = newAnnotations[filename];
            }});

            // Convert back to sorted array
            const merged = Object.keys(existingByFilename)
                .sort()
                .map(filename => existingByFilename[filename]);

            const json = JSON.stringify(merged, null, 2);
            const blob = new Blob([json], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ground_truth_annotations.json';
            a.click();
            URL.revokeObjectURL(url);

            alert(
                'Ground truth exported!\\n\\n' +
                'New annotations: ' + Object.keys(newAnnotations).length + '\\n' +
                'Total annotations: ' + merged.length + '\\n\\n' +
                'Copy the downloaded file to:\\n' +
                'test_real_images/ground_truth_annotations.json'
            );
        }}

        function generateResults() {{
            const stats = {{
                images_with_manual: 0,
                images_with_detections: 0,
                total_manual_boxes: 0,
                total_selected_boxes: 0
            }};

            imageData.forEach(data => {{
                const selected = selections[data.idx];
                const manual = manualBoxes[data.idx];

                if (manual.length > 0) {{
                    stats.images_with_manual++;
                    stats.total_manual_boxes += manual.length;
                }}

                if (selected.length > 0) {{
                    stats.images_with_detections++;
                    stats.total_selected_boxes += selected.length;
                }}
            }});

            let output = 'GROUND TRUTH ANNOTATION SUMMARY\\n';
            output += '='.repeat(50) + '\\n\\n';
            output += `Total images shown: ${{imageData.length}}\\n`;
            output += `Images with selected detections: ${{stats.images_with_detections}}\\n`;
            output += `Images with manual boxes: ${{stats.images_with_manual}}\\n`;
            output += `Total selected detection boxes: ${{stats.total_selected_boxes}}\\n`;
            output += `Total manual boxes: ${{stats.total_manual_boxes}}\\n`;

            const unannotated = imageData.length - imageData.filter(d =>
                selections[d.idx].length > 0 || manualBoxes[d.idx].length > 0
            ).length;
            output += `\\nUnannotated images: ${{unannotated}}\\n`;

            document.getElementById('results').textContent = output;
            document.getElementById('results').style.display = 'block';
            document.getElementById('results').scrollIntoView({{ behavior: 'smooth' }});
        }}

        function saveProgress() {{
            const data = JSON.stringify({{ selections, manualBoxes }});
            localStorage.setItem('annotate_ground_truth_data', data);
            alert('Progress saved to browser!');
        }}

        function loadProgress() {{
            const data = localStorage.getItem('annotate_ground_truth_data');
            if (!data) {{
                alert('No saved progress found');
                return;
            }}
            const loaded = JSON.parse(data);
            Object.assign(selections, loaded.selections);
            Object.assign(manualBoxes, loaded.manualBoxes);
            imageData.forEach(data => {{
                drawDetections(data.idx);
                updateDetectionList(data.idx);
            }});
            updateProgress();
            alert('Progress loaded!');
        }}

        initImages();
    </script>
</body>
</html>
"""

    # Save HTML
    output_path = output_dir / 'annotate_ground_truth.html'
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nDone! Annotation tool saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Open in browser:  file://{output_path.absolute()}")
    print(f"  2. For each image, either:")
    print(f"     - Click on ensemble detection boxes to mark them as correct")
    print(f"     - Click 'Draw Box' and draw manual ground truth boxes")
    print(f"  3. Click 'Export Ground Truth JSON' to download annotations")
    print(f"  4. Copy the downloaded file to: {GROUND_TRUTH_PATH}")
    if mode != 'all':
        print(f"\n  Note: Exported JSON will be merged with {len(existing_annotations)} existing annotations.")


if __name__ == '__main__':
    main()
