#!/usr/bin/env python3
"""Generate interactive HTML report for testing the updated preprocessor with feedback capability."""

import json
import sys
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frame_prep.detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from frame_prep.cropper import SmartCropper


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def draw_boxes_on_image(image_path, detections, ground_truth_boxes=None, max_width=800):
    """Draw detected and ground truth bounding boxes on image."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Handle EXIF rotation
        img = ImageOps.exif_transpose(img)

        scale = 1.0
        if img.width > max_width:
            scale = max_width / img.width
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Draw ground truth boxes in blue
        if ground_truth_boxes:
            for gt_box in ground_truth_boxes:
                bbox = [int(coord * scale) for coord in gt_box]
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
                draw.text((x1, y2 + 5), "Ground Truth", fill=(0, 0, 255), font=small_font)

        # Draw detected boxes in green
        if detections:
            primary_det = detections[0]  # First detection is primary
            for i, det in enumerate(detections[:5]):  # Show up to 5 detections
                bbox = [int(coord * scale) for coord in det.bbox]
                x1, y1, x2, y2 = bbox

                # Primary detection gets thicker border
                width = 4 if i == 0 else 2
                color = (0, 255, 0) if i == 0 else (0, 200, 0)

                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

                label = f"{det.class_name} {det.confidence:.2f}"
                if i == 0:
                    label = "PRIMARY: " + label

                text_bbox = draw.textbbox((x1, y1-20), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                             fill=color)
                draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"Error drawing boxes on {image_path}: {e}")
        return None


def run_detection(image_path, detector):
    """Run detection on an image and return results."""
    try:
        img = Image.open(image_path)
        # Handle EXIF rotation
        img = ImageOps.exif_transpose(img)

        # Pass image_path for cache lookups
        try:
            detections = detector.detect(img, verbose=False, image_path=image_path)
        except TypeError:
            detections = detector.detect(img, verbose=False)

        primary = detector.get_primary_subject(detections) if detections else None

        return {
            'all_detections': detections,
            'primary': primary,
            'count': len(detections)
        }
    except Exception as e:
        print(f"Error detecting in {image_path}: {e}")
        return {'all_detections': [], 'primary': None, 'count': 0}


def check_accuracy(detections, ground_truth_boxes, iou_threshold=0.3):
    """Check if detection matches ground truth."""
    if not detections or not ground_truth_boxes:
        return False, 0.0

    # Check if primary detection overlaps with any ground truth box
    primary = detections[0]
    best_iou = 0.0

    for gt_box in ground_truth_boxes:
        iou = calculate_iou(primary.bbox, gt_box)
        best_iou = max(best_iou, iou)

    return best_iou >= iou_threshold, best_iou


def generate_report():
    """Generate interactive HTML report."""
    print("Loading ground truth annotations...")
    with open('test_real_images/ground_truth_annotations.json', 'r') as f:
        ground_truth = json.load(f)

    input_dir = Path('test_real_images/input')
    results = []

    # Create detector once (reused for all images, with caching)
    detector = OptimizedEnsembleDetector(confidence_threshold=0.25, merge_threshold=0.2)

    print(f"\nProcessing {len(ground_truth)} test images...")

    correct_count = 0
    total_with_gt = 0

    for idx, gt_entry in enumerate(ground_truth, 1):
        filename = gt_entry['filename']
        image_path = input_dir / filename

        if not image_path.exists():
            print(f"  [{idx}/{len(ground_truth)}] Skipping missing: {filename}")
            continue

        print(f"  [{idx}/{len(ground_truth)}] Processing: {filename}")

        # Get ground truth boxes
        gt_boxes = []
        for box_data in gt_entry.get('manual_boxes', []):
            gt_boxes.append(box_data['bbox'])
        for det_data in gt_entry.get('correct_detections', []):
            gt_boxes.append(det_data['bbox'])

        # Run detection with optimized ensemble (uses caching)
        detection_result = run_detection(image_path, detector)

        # Check accuracy
        is_correct = False
        best_iou = 0.0
        if gt_boxes and detection_result['all_detections']:
            is_correct, best_iou = check_accuracy(detection_result['all_detections'], gt_boxes)
            total_with_gt += 1
            if is_correct:
                correct_count += 1

        # Generate visualization
        img_with_boxes = draw_boxes_on_image(
            image_path,
            detection_result['all_detections'],
            gt_boxes
        )

        results.append({
            'filename': filename,
            'image_with_boxes': img_with_boxes,
            'detections': detection_result['all_detections'],
            'primary': detection_result['primary'],
            'detection_count': detection_result['count'],
            'has_ground_truth': len(gt_boxes) > 0,
            'is_correct': is_correct,
            'best_iou': best_iou,
            'ground_truth_boxes': gt_boxes
        })

    accuracy = (correct_count / total_with_gt * 100) if total_with_gt > 0 else 0

    print(f"\n✓ Processing complete!")
    print(f"  Accuracy: {correct_count}/{total_with_gt} ({accuracy:.1f}%)")
    print(f"\nGenerating HTML report...")

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Detection Report - Updated Zoom Logic</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 16px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
        }}

        .stat-value {{
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
            line-height: 1;
        }}

        .stat-label {{
            font-size: 14px;
            color: #6b7280;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .filters {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .filter-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-btn {{
            padding: 10px 20px;
            border: 2px solid #e5e7eb;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            font-weight: 500;
        }}

        .filter-btn:hover {{
            border-color: #667eea;
            color: #667eea;
        }}

        .filter-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .results-grid {{
            display: grid;
            gap: 25px;
        }}

        .result-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }}

        .result-card.correct {{
            border-left: 5px solid #10b981;
        }}

        .result-card.incorrect {{
            border-left: 5px solid #ef4444;
        }}

        .result-card.no-gt {{
            border-left: 5px solid #6b7280;
        }}

        .result-header {{
            padding: 20px;
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
        }}

        .result-title {{
            font-size: 16px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 10px;
        }}

        .result-meta {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            font-size: 13px;
            color: #6b7280;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .badge.correct {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge.incorrect {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .badge.no-gt {{
            background: #e5e7eb;
            color: #374151;
        }}

        .result-image {{
            width: 100%;
            display: block;
        }}

        .result-details {{
            padding: 20px;
        }}

        .detection-info {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}

        .detection-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .detection-value {{
            font-size: 14px;
            color: #1f2937;
            line-height: 1.5;
        }}

        .feedback-section {{
            padding-top: 15px;
            border-top: 1px solid #e5e7eb;
        }}

        .feedback-label {{
            font-size: 13px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 10px;
        }}

        .feedback-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .feedback-btn {{
            flex: 1;
            padding: 10px;
            border: 2px solid #e5e7eb;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
        }}

        .feedback-btn:hover {{
            border-color: #667eea;
        }}

        .feedback-btn.selected {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .feedback-textarea {{
            width: 100%;
            padding: 10px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 60px;
        }}

        .feedback-textarea:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .legend h3 {{
            font-size: 16px;
            margin-bottom: 15px;
            color: #1f2937;
        }}

        .legend-items {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .legend-color {{
            width: 30px;
            height: 4px;
            border-radius: 2px;
        }}

        .export-btn {{
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .export-btn:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Interactive Detection Report</h1>
        <p>Updated Adaptive Zoom Logic - {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{accuracy:.1f}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{correct_count}/{total_with_gt}</div>
            <div class="stat-label">Correct Detections</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(results)}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{sum(1 for r in results if r['detection_count'] > 0)}</div>
            <div class="stat-label">Images with Detections</div>
        </div>
    </div>

    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-color" style="background: #10b981; width: 40px; height: 4px;"></div>
                <span>Primary Detection (GREEN - thick border)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #0ea5e9; width: 40px; height: 4px;"></div>
                <span>Ground Truth (BLUE)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #6ee7b7; width: 40px; height: 4px;"></div>
                <span>Other Detections (light green)</span>
            </div>
        </div>
    </div>

    <div class="filters">
        <div class="filter-group">
            <button class="filter-btn active" onclick="filterResults('all')">All ({len(results)})</button>
            <button class="filter-btn" onclick="filterResults('correct')">Correct ({sum(1 for r in results if r['is_correct'])})</button>
            <button class="filter-btn" onclick="filterResults('incorrect')">Incorrect ({sum(1 for r in results if r['has_ground_truth'] and not r['is_correct'])})</button>
            <button class="filter-btn" onclick="filterResults('no-gt')">No Ground Truth ({sum(1 for r in results if not r['has_ground_truth'])})</button>
            <button class="export-btn" onclick="exportFeedback()">📥 Export Feedback</button>
        </div>
    </div>

    <div class="results-grid" id="results-grid">
"""

    # Add each result
    for idx, result in enumerate(results):
        status_class = 'no-gt' if not result['has_ground_truth'] else ('correct' if result['is_correct'] else 'incorrect')
        status_label = 'No Ground Truth' if not result['has_ground_truth'] else ('✓ Correct' if result['is_correct'] else '✗ Incorrect')

        primary_info = "No detections"
        if result['primary']:
            primary_info = f"{result['primary'].class_name} (conf: {result['primary'].confidence:.3f})"

        iou_info = ""
        if result['has_ground_truth'] and result['detection_count'] > 0:
            iou_info = f"<span>IoU: {result['best_iou']:.3f}</span>"

        html += f"""
        <div class="result-card {status_class}" data-status="{status_class}" data-index="{idx}">
            <div class="result-header">
                <div class="result-title">{result['filename']}</div>
                <div class="result-meta">
                    <span class="badge {status_class}">{status_label}</span>
                    <span>Detections: {result['detection_count']}</span>
                    {iou_info}
                </div>
            </div>

            <img src="{result['image_with_boxes']}" class="result-image" alt="{result['filename']}">

            <div class="result-details">
                <div class="detection-info">
                    <div class="detection-label">Primary Detection</div>
                    <div class="detection-value">{primary_info}</div>
                </div>

                <div class="feedback-section">
                    <div class="feedback-label">Your Feedback:</div>
                    <div class="feedback-buttons">
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'good')">
                            👍 Good Detection
                        </button>
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'bad')">
                            👎 Poor Detection
                        </button>
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'zoom')">
                            🔍 Zoom Issue
                        </button>
                    </div>
                    <textarea
                        class="feedback-textarea"
                        placeholder="Optional: Add comments about detection quality, zoom appropriateness, or subject identification..."
                        onchange="updateComment({idx}, this.value)"
                    ></textarea>
                </div>
            </div>
        </div>
"""

    html += """
    </div>

    <script>
        const feedbackData = {};

        function filterResults(filter) {
            const cards = document.querySelectorAll('.result-card');
            const buttons = document.querySelectorAll('.filter-btn');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            cards.forEach(card => {
                if (filter === 'all') {
                    card.style.display = 'block';
                } else {
                    card.style.display = card.dataset.status === filter ? 'block' : 'none';
                }
            });
        }

        function setFeedback(index, rating) {
            if (!feedbackData[index]) {
                feedbackData[index] = {};
            }
            feedbackData[index].rating = rating;

            // Update button states
            const card = document.querySelector(`[data-index="${index}"]`);
            const buttons = card.querySelectorAll('.feedback-btn');
            buttons.forEach(btn => btn.classList.remove('selected'));

            const ratingMap = {
                'good': 0,
                'bad': 1,
                'zoom': 2
            };
            buttons[ratingMap[rating]].classList.add('selected');
        }

        function updateComment(index, comment) {
            if (!feedbackData[index]) {
                feedbackData[index] = {};
            }
            feedbackData[index].comment = comment;
        }

        function exportFeedback() {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `detection_feedback_${timestamp}.json`;

            const exportData = {
                generated_at: new Date().toISOString(),
                total_images: """ + str(len(results)) + """,
                feedback_count: Object.keys(feedbackData).length,
                feedback: feedbackData
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);

            alert(`Exported feedback for ${Object.keys(feedbackData).length} images`);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'f' && e.ctrlKey) {
                e.preventDefault();
                exportFeedback();
            }
        });
    </script>
</body>
</html>
"""

    # Save report
    output_path = Path('reports/interactive_detection_report.html')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✓ Report generated: {output_path}")
    print(f"\nOpen in browser to view and provide feedback:")
    print(f"  file://{output_path.absolute()}")
    print(f"\nKeyboard shortcuts:")
    print(f"  Ctrl+F: Export feedback to JSON")


if __name__ == '__main__':
    generate_report()
