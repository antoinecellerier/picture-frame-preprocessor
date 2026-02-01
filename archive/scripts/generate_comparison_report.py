#!/usr/bin/env python3
"""Generate HTML comparison report for all detection models."""

import json
import os
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont
import io

def image_to_base64(image_path, max_width=600):
    """Convert image to base64 for HTML embedding."""
    try:
        from PIL import ImageOps
        img = Image.open(image_path)

        # Fix rotation based on EXIF
        img = ImageOps.exif_transpose(img)

        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def draw_boxes_on_image(image_path, detections, color=(0, 255, 0), max_width=600):
    """Draw bounding boxes on image and return base64."""
    try:
        from PIL import ImageOps
        img = Image.open(image_path).convert('RGB')

        # Fix rotation based on EXIF first
        img = ImageOps.exif_transpose(img)

        # Resize if needed
        scale = 1.0
        if img.width > max_width:
            scale = max_width / img.width
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        draw = ImageDraw.Draw(img)

        # Draw each detection
        for det in detections[:10]:  # Limit to top 10
            bbox = det['bbox']
            # Scale bbox if image was resized
            bbox = [int(coord * scale) for coord in bbox]
            x1, y1, x2, y2 = bbox

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = ImageFont.load_default()

            # Background for text
            text_bbox = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        return None

def generate_comparison_report():
    """Generate HTML comparison report."""

    # Load ground truth
    with open('test_real_images/ground_truth_annotations.json', 'r') as f:
        ground_truth = json.load(f)

    # Load caches
    caches = {}
    cache_files = {
        'YOLOv8m': 'cache/yolov8m_conf0.15.json',
        'RT-DETR-L': 'cache/rtdetr-l_conf0.15.json',
        'YOLO-World': 'cache/yolo_world_conf0.25.json',
        'Grounding DINO': 'cache/grounding_dino_conf0.25.json'
    }

    for name, path in cache_files.items():
        if Path(path).exists():
            with open(path, 'r') as f:
                caches[name] = json.load(f)

    input_dir = Path('test_real_images/input')

    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Comparison - Art Detection Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 20px 0;
            color: #333;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .summary-card.winner {
            border-left-color: #28a745;
            background: #d4edda;
        }
        .summary-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .summary-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .summary-subtitle {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .comparison-grid {
            display: grid;
            gap: 30px;
        }
        .image-comparison {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .comparison-header {
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
            background: #f8f9fa;
        }
        .filename {
            font-weight: 600;
            font-size: 16px;
            color: #333;
            margin-bottom: 10px;
        }
        .status-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .badge {
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge.success {
            background: #d4edda;
            color: #155724;
        }
        .badge.fail {
            background: #f8d7da;
            color: #721c24;
        }
        .badge.partial {
            background: #fff3cd;
            color: #856404;
        }
        .images-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2px;
            background: #e9ecef;
        }
        .model-result {
            background: white;
            padding: 15px;
        }
        .model-name {
            font-weight: 600;
            font-size: 14px;
            color: #333;
            margin-bottom: 5px;
        }
        .detection-count {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
        }
        .model-result img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 4px;
            border: 2px solid #e9ecef;
        }
        .winner-badge {
            background: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Art Detection Model Comparison</h1>
        <p style="color: #666; margin: 5px 0 20px 0;">Evaluation on 63 annotated museum/gallery images with 142 ground truth bounding boxes</p>

        <div class="summary">
            <div class="summary-card">
                <div class="summary-label">YOLOv8m (Baseline)</div>
                <div class="summary-value">38.1%</div>
                <div class="summary-subtitle">24/63 images detected</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">YOLO-World ⚡</div>
                <div class="summary-value">52.4%</div>
                <div class="summary-subtitle">33/63 images • 0.7s/img</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Ensemble (YOLO+DETR)</div>
                <div class="summary-value">63.5%</div>
                <div class="summary-subtitle">40/63 images • 3.5s/img</div>
            </div>
            <div class="summary-card winner">
                <div class="summary-label">🏆 Grounding DINO</div>
                <div class="summary-value">88.9%</div>
                <div class="summary-subtitle">56/63 images • 12s/img</div>
            </div>
        </div>
    </div>

    <div class="comparison-grid">
"""

    # Process each image
    for gt_entry in ground_truth[:20]:  # Limit to first 20 for reasonable HTML size
        filename = gt_entry['filename']
        img_path = input_dir / filename

        if not img_path.exists():
            continue

        # Get ground truth boxes
        gt_boxes = []
        for box_data in gt_entry.get('manual_boxes', []):
            gt_boxes.append(box_data['bbox'])
        for det_data in gt_entry.get('correct_detections', []):
            gt_boxes.append(det_data['bbox'])

        has_gt = len(gt_boxes) > 0

        # Get detections from each model
        model_results = {}
        for model_name, cache_data in caches.items():
            if filename in cache_data['detections']:
                detections = cache_data['detections'][filename]['detections']
                model_results[model_name] = detections
            else:
                model_results[model_name] = []

        # Determine success for each model
        statuses = {}
        for model_name, detections in model_results.items():
            if has_gt:
                statuses[model_name] = 'success' if len(detections) > 0 else 'fail'
            else:
                statuses[model_name] = 'unknown'

        # Create HTML for this image
        html += f"""
        <div class="image-comparison">
            <div class="comparison-header">
                <div class="filename">{filename}</div>
                <div class="status-badges">
                    <span class="badge {'success' if statuses.get('YOLOv8m') == 'success' else 'fail'}">
                        YOLOv8m: {len(model_results.get('YOLOv8m', []))} det
                    </span>
                    <span class="badge {'success' if statuses.get('YOLO-World') == 'success' else 'fail'}">
                        YOLO-World: {len(model_results.get('YOLO-World', []))} det
                    </span>
                    <span class="badge {'success' if statuses.get('Grounding DINO') == 'success' else 'fail'}">
                        Grounding DINO: {len(model_results.get('Grounding DINO', []))} det
                        {' <span class="winner-badge">🏆</span>' if statuses.get('Grounding DINO') == 'success' else ''}
                    </span>
                </div>
            </div>

            <div class="images-row">
"""

        # Original image
        orig_b64 = image_to_base64(img_path)
        if orig_b64:
            html += f"""
                <div class="model-result">
                    <div class="model-name">Original</div>
                    <div class="detection-count">{len(gt_boxes)} ground truth boxes</div>
                    <img src="{orig_b64}" alt="Original">
                </div>
"""

        # YOLOv8m
        if 'YOLOv8m' in model_results:
            yolo_img = draw_boxes_on_image(img_path, model_results['YOLOv8m'], color=(255, 0, 0))
            if yolo_img:
                html += f"""
                <div class="model-result">
                    <div class="model-name">YOLOv8m <span style="color: red;">(red)</span></div>
                    <div class="detection-count">{len(model_results['YOLOv8m'])} detections</div>
                    <img src="{yolo_img}" alt="YOLOv8m">
                </div>
"""

        # YOLO-World
        if 'YOLO-World' in model_results:
            yolo_world_img = draw_boxes_on_image(img_path, model_results['YOLO-World'], color=(0, 150, 255))
            if yolo_world_img:
                html += f"""
                <div class="model-result">
                    <div class="model-name">YOLO-World ⚡ <span style="color: #0096ff;">(blue)</span></div>
                    <div class="detection-count">{len(model_results['YOLO-World'])} detections</div>
                    <img src="{yolo_world_img}" alt="YOLO-World">
                </div>
"""

        # Grounding DINO
        if 'Grounding DINO' in model_results:
            dino_img = draw_boxes_on_image(img_path, model_results['Grounding DINO'], color=(0, 255, 0))
            if dino_img:
                html += f"""
                <div class="model-result">
                    <div class="model-name">Grounding DINO 🏆 <span style="color: green;">(green)</span></div>
                    <div class="detection-count">{len(model_results['Grounding DINO'])} detections</div>
                    <img src="{dino_img}" alt="Grounding DINO">
                </div>
"""

        html += """
            </div>
        </div>
"""

    html += """
    </div>

    <div style="text-align: center; padding: 40px; color: #666;">
        <p>Showing first 20 images with bounding boxes:</p>
        <p><span style="color: red;">■</span> YOLOv8m (red) • <span style="color: #0096ff;">■</span> YOLO-World (blue) • <span style="color: green;">■</span> Grounding DINO (green)</p>
        <p style="margin-top: 20px;"><strong>🏆 Grounding DINO: 88.9% accuracy (winner!)</strong></p>
        <p><strong>⚡ YOLO-World: 52.4% accuracy, 17x faster</strong></p>
        <p><strong>📊 YOLOv8m: 38.1% accuracy (baseline)</strong></p>
    </div>
</body>
</html>
"""

    # Save report
    report_path = 'model_comparison_report.html'
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"✓ Report generated: {report_path}")
    print(f"  Open in browser: file://{os.path.abspath(report_path)}")

if __name__ == '__main__':
    generate_comparison_report()
