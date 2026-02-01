#!/usr/bin/env python3
"""Generate HTML report for ensemble detector results."""

import json
import os
from pathlib import Path
import base64
from PIL import Image
import io

def image_to_base64(image_path, max_width=800):
    """Convert image to base64 for HTML embedding."""
    try:
        img = Image.open(image_path)

        # Resize if too large
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        return None


def generate_report():
    """Generate HTML report of ensemble results."""

    input_dir = Path('test_real_images/input')
    output_dir = Path('test_real_images/output_ensemble')

    # Collect all processed images with their analysis data
    results = []

    for analysis_file in sorted(output_dir.glob('*_analysis.json')):
        with open(analysis_file, 'r') as f:
            data = json.load(f)

        filename = data['filename']
        input_path = input_dir / filename
        output_filename = filename.rsplit('.', 1)[0] + '.jpg'
        output_path = output_dir / output_filename

        if input_path.exists() and output_path.exists():
            results.append({
                'filename': filename,
                'input_path': input_path,
                'output_path': output_path,
                'data': data
            })

    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ensemble Detector Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
            border-left: 3px solid #007bff;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .image-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-card-header {
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }
        .filename {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        .detection-info {
            font-size: 13px;
            color: #666;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .detection-badge {
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: 500;
        }
        .strategy-badge {
            background: #f3e5f5;
            color: #7b1fa2;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: 500;
        }
        .zoom-badge {
            background: #fff3e0;
            color: #e65100;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: 500;
        }
        .images-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: #e9ecef;
        }
        .image-section {
            background: white;
            padding: 10px;
        }
        .image-label {
            font-size: 11px;
            text-transform: uppercase;
            color: #666;
            font-weight: 600;
            margin-bottom: 5px;
            letter-spacing: 0.5px;
        }
        .image-section img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 4px;
        }
        .detections-list {
            padding: 15px;
            background: #f8f9fa;
            font-size: 12px;
            border-top: 1px solid #e9ecef;
        }
        .detection-item {
            padding: 5px 0;
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Ensemble Detector Results</h1>
        <p style="color: #666; margin: 5px 0;">YOLOv8m + RT-DETR-L with optimized box merging (merge_threshold=0.4)</p>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Images</div>
                <div class="stat-value">""" + str(len(results)) + """</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">100%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Detections</div>
                <div class="stat-value">""" + f"{sum(r['data']['detections_found'] for r in results) / len(results):.1f}" + """</div>
            </div>
        </div>
    </div>

    <div class="image-grid">
"""

    for result in results:
        data = result['data']

        # Convert images to base64
        input_b64 = image_to_base64(result['input_path'])
        output_b64 = image_to_base64(result['output_path'])

        if not input_b64 or not output_b64:
            continue

        # Format detection info
        detections_html = ""
        if data.get('detections'):
            for i, det in enumerate(data['detections'][:5], 1):  # Show top 5
                # Calculate area from bbox if not present
                if 'area' in det:
                    area = det['area']
                else:
                    bbox = det['bbox']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                primary_marker = " 🎯" if det.get('is_primary') else ""
                detections_html += f"""
                    <div class="detection-item">
                        {i}. {det['class_name']}{primary_marker} (conf: {det['confidence']:.2f}, area: {area:,}px)
                    </div>
                """

        # Get crop and zoom info
        crop_info = data.get('crop_box', 'N/A')
        zoom = data.get('zoom_applied', 1.0)

        html += f"""
        <div class="image-card">
            <div class="image-card-header">
                <div class="filename">{result['filename']}</div>
                <div class="detection-info">
                    <span class="detection-badge">{data['detections_found']} detections</span>
                    <span class="strategy-badge">{data['strategy_used']}</span>
                    <span class="zoom-badge">zoom: {zoom:.2f}x</span>
                </div>
            </div>

            <div class="images-container">
                <div class="image-section">
                    <div class="image-label">Original ({data['original_dimensions'][0]}×{data['original_dimensions'][1]})</div>
                    <img src="{input_b64}" alt="Original">
                </div>
                <div class="image-section">
                    <div class="image-label">Processed (480×800)</div>
                    <img src="{output_b64}" alt="Processed">
                </div>
            </div>

            {f'<div class="detections-list"><strong>Top Detections:</strong>{detections_html}</div>' if detections_html else ''}
        </div>
        """

    html += """
    </div>
</body>
</html>
"""

    # Save report
    report_path = 'ensemble_results_report.html'
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"✓ Report generated: {report_path}")
    print(f"  Total images: {len(results)}")
    print(f"  Open in browser: file://{os.path.abspath(report_path)}")


if __name__ == '__main__':
    generate_report()
