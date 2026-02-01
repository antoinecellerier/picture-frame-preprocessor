#!/usr/bin/env python3
"""Generate interactive HTML quality assessment report."""

import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from frame_prep.detector import ArtFeatureDetector
from frame_prep.cropper import SmartCropper


def image_to_base64(image_path, max_height=None):
    """Convert image to base64 for embedding in HTML."""
    from PIL import ImageOps

    with Image.open(image_path) as img:
        # Apply EXIF orientation (fixes rotated images)
        img = ImageOps.exif_transpose(img)

        if max_height and img.height > max_height:
            ratio = max_height / img.height
            new_size = (int(img.width * ratio), max_height)
            img = img.resize(new_size, Image.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"


def analyze_image_smart(input_path, output_path, target_width=480, target_height=800):
    """
    Analyze image with ML detection and get crop info.
    First checks for cached JSON analysis, otherwise runs ML detection.

    Args:
        input_path: Path to original image
        output_path: Path to processed image (to find JSON)
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns dict with:
        - original: "WxH" dimensions
        - detections: List of {bbox, confidence, class_name, is_primary}
        - crop_box: (left, top, right, bottom) in pixels
        - crop_box_pct: (left%, top%, width%, height%) for CSS
    """
    from PIL import ImageOps

    # Check for cached analysis JSON
    json_path = output_path.rsplit('.', 1)[0] + '_analysis.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                analysis_data = json.load(f)

            # Get original dimensions
            width, height = analysis_data.get('original_dimensions', (0, 0))

            # Use cached crop box (handle None for portrait images)
            crop_box_data = analysis_data.get('crop_box')
            if crop_box_data is None:
                # Portrait image - no crop needed
                crop_box = (0, 0, width, height)
            else:
                crop_box = tuple(crop_box_data)

            # Convert to percentages for CSS
            crop_box_pct = (
                (crop_box[0] / width) * 100 if width > 0 else 0,
                (crop_box[1] / height) * 100 if height > 0 else 0,
                ((crop_box[2] - crop_box[0]) / width) * 100 if width > 0 else 100,
                ((crop_box[3] - crop_box[1]) / height) * 100 if height > 0 else 100
            )

            # Format detections for visualization
            detection_info = []
            for det in analysis_data.get('detections', []):
                bbox = tuple(det['bbox'])
                detection_info.append({
                    'bbox': bbox,
                    'bbox_pct': (
                        (bbox[0] / width) * 100 if width > 0 else 0,
                        (bbox[1] / height) * 100 if height > 0 else 0,
                        ((bbox[2] - bbox[0]) / width) * 100 if width > 0 else 0,
                        ((bbox[3] - bbox[1]) / height) * 100 if height > 0 else 0
                    ),
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'is_primary': det['is_primary']
                })

            return {
                'original': f"{width}x{height}",
                'detections': detection_info,
                'crop_box': crop_box,
                'crop_box_pct': crop_box_pct,
                'detection_count': len(detection_info)
            }

        except Exception as e:
            # Fall through to re-analyze if JSON read fails
            print(f"Warning: Failed to read cached analysis for {input_path}: {e}")

    # No cached analysis - run ML detection
    with Image.open(input_path) as img:
        # Apply EXIF orientation
        img = ImageOps.exif_transpose(img)
        width, height = img.size

        # Initialize detector and cropper with improved settings
        detector = ArtFeatureDetector(model_name='yolov8m', confidence_threshold=0.15)
        cropper = SmartCropper(target_width, target_height, zoom_factor=1.3, use_saliency_fallback=True)

        # Run detection
        detections = detector.detect(img, verbose=False)

        # Get primary subject
        primary_detection = detector.get_primary_subject(detections) if detections else None

        # Calculate crop window
        if cropper.needs_cropping(img):
            if detections:
                # Smart crop based on primary detection
                primary = detections[0]
                anchor_x, anchor_y = primary.center
            else:
                # Fallback to center
                anchor_x = width // 2
                anchor_y = height // 2

            # Calculate crop window (same logic as cropper)
            target_aspect = target_height / target_width
            crop_width = height / target_aspect
            crop_height = height

            left = anchor_x - crop_width / 2
            right = left + crop_width
            top = 0
            bottom = height

            # Clamp to bounds
            if left < 0:
                left = 0
                right = crop_width
            if right > width:
                right = width
                left = width - crop_width

            left = max(0, left)
            right = min(width, right)

            crop_box = (int(left), int(top), int(right), int(bottom))
        else:
            # No crop needed
            crop_box = (0, 0, width, height)

        # Convert to percentages for CSS
        crop_box_pct = (
            (crop_box[0] / width) * 100,
            (crop_box[1] / height) * 100,
            ((crop_box[2] - crop_box[0]) / width) * 100,
            ((crop_box[3] - crop_box[1]) / height) * 100
        )

        # Format detections for visualization
        detection_info = []
        for det in detections:
            detection_info.append({
                'bbox': det.bbox,
                'bbox_pct': (
                    (det.bbox[0] / width) * 100,
                    (det.bbox[1] / height) * 100,
                    ((det.bbox[2] - det.bbox[0]) / width) * 100,
                    ((det.bbox[3] - det.bbox[1]) / height) * 100
                ),
                'confidence': det.confidence,
                'class_name': det.class_name,
                'is_primary': det == primary_detection
            })

        return {
            'original': f"{width}x{height}",
            'detections': detection_info,
            'crop_box': crop_box,
            'crop_box_pct': crop_box_pct,
            'detection_count': len(detections)
        }


def generate_html_report(input_dir, output_dir, html_path, title="Crop Quality Assessment"):
    """
    Generate interactive HTML quality assessment report.

    Args:
        input_dir: Directory with original images
        output_dir: Directory with processed images
        html_path: Path to save HTML report
        title: Report title
    """
    print(f"Generating quality report...")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #1a1a1a;
            font-size: 28px;
        }}
        .info {{
            color: #6c757d;
            font-size: 14px;
            line-height: 1.6;
        }}
        .stats-bar {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #28a745;
        }}
        .stat-label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        .comparison {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: box-shadow 0.2s;
        }}
        .comparison:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }}
        .comparison-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .filename {{
            font-weight: 600;
            font-size: 16px;
            color: #1a1a1a;
        }}
        .image-number {{
            background: #e9ecef;
            color: #495057;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 600;
        }}
        .images-container {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .image-box {{
            min-width: 0;
        }}
        .image-label {{
            font-weight: 600;
            margin-bottom: 10px;
            color: #495057;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .image-wrapper {{
            position: relative;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            background: #f8f9fa;
        }}
        .image-wrapper img {{
            display: block;
            width: 100%;
            height: auto;
        }}
        .ml-overlay {{
            position: absolute;
            pointer-events: none;
        }}
        .detection-box {{
            position: absolute;
            border: 2px solid #ffc107;
            background: rgba(255, 193, 7, 0.1);
            pointer-events: none;
        }}
        .detection-box.primary {{
            border: 3px solid #28a745;
            background: rgba(40, 167, 69, 0.15);
            box-shadow: 0 0 10px rgba(40, 167, 69, 0.5);
        }}
        .detection-label {{
            position: absolute;
            top: -24px;
            left: 0;
            background: rgba(255, 193, 7, 0.95);
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            white-space: nowrap;
        }}
        .detection-label.primary {{
            background: rgba(40, 167, 69, 0.95);
        }}
        .crop-area {{
            position: absolute;
            border: 3px solid #17a2b8;
            background: rgba(23, 162, 184, 0.08);
            pointer-events: none;
            box-shadow: 0 0 0 2000px rgba(0, 0, 0, 0.3);
        }}
        .crop-label {{
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(23, 162, 184, 0.95);
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 0.5px;
        }}
        .ml-info {{
            margin-top: 8px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 11px;
        }}
        .ml-info .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            margin-right: 4px;
        }}
        .ml-info .badge.primary {{
            background: #28a745;
            color: white;
        }}
        .ml-info .badge.detected {{
            background: #ffc107;
            color: #333;
        }}
        .stats {{
            margin-top: 8px;
            font-size: 12px;
            color: #6c757d;
        }}
        .feedback-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
        }}
        .feedback-title {{
            font-weight: 600;
            margin-bottom: 15px;
            color: #1a1a1a;
            font-size: 14px;
        }}
        .rating-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .rating-btn {{
            flex: 1;
            padding: 12px;
            border: 2px solid #dee2e6;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s;
            text-align: center;
        }}
        .rating-btn:hover {{
            border-color: #adb5bd;
            transform: translateY(-2px);
        }}
        .rating-btn.excellent {{
            color: #28a745;
        }}
        .rating-btn.good {{
            color: #20c997;
        }}
        .rating-btn.fair {{
            color: #ffc107;
        }}
        .rating-btn.poor {{
            color: #fd7e14;
        }}
        .rating-btn.bad {{
            color: #dc3545;
        }}
        .rating-btn.selected {{
            border-width: 3px;
            transform: scale(1.05);
        }}
        .rating-btn.excellent.selected {{
            background: #d4edda;
            border-color: #28a745;
        }}
        .rating-btn.good.selected {{
            background: #d1ecf1;
            border-color: #20c997;
        }}
        .rating-btn.fair.selected {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        .rating-btn.poor.selected {{
            background: #ffe5d0;
            border-color: #fd7e14;
        }}
        .rating-btn.bad.selected {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        .comment-box {{
            width: 100%;
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 13px;
            font-family: inherit;
            resize: vertical;
            min-height: 60px;
        }}
        .comment-box:focus {{
            outline: none;
            border-color: #80bdff;
        }}
        .legend {{
            background: #d1ecf1;
            padding: 15px 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            border-left: 4px solid #17a2b8;
            font-size: 14px;
            line-height: 1.6;
        }}
        .legend strong {{
            color: #dc3545;
        }}
        .export-section {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            z-index: 1000;
        }}
        .export-btn {{
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .export-btn:hover {{
            background: #218838;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .feedback-count {{
            font-size: 12px;
            color: #6c757d;
            margin-top: 8px;
            text-align: center;
        }}
        @media (max-width: 768px) {{
            .images-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 {title}</h1>
        <div class="info">
            <strong>Strategy:</strong> Smart (ML-based with YOLOv8)<br>
            <strong>Target Output:</strong> 480x800 portrait<br>
            <strong>Instructions:</strong> Review each crop and rate the quality. Red boxes show the cropped area.
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat">
            <div class="stat-value" id="totalCount">0</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="ratedCount">0</div>
            <div class="stat-label">Rated</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="avgRating">-</div>
            <div class="stat-label">Avg Rating</div>
        </div>
    </div>

    <div class="legend">
        <strong>ML Visualization:</strong>
        <span style="color: #28a745; font-weight: bold;">⬛ Green boxes</span> = Primary subject (used for crop anchor) •
        <span style="color: #ffc107; font-weight: bold;">⬛ Yellow boxes</span> = Other detected objects •
        <span style="color: #17a2b8; font-weight: bold;">⬛ Blue area</span> = Smart crop region (becomes the 480x800 output)
        <br>
        Rate whether the crop preserves important subjects and composition.
    </div>
"""

    # Process images
    input_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])

    total_images = len(input_files)
    ml_analysis_cache = {}  # Store ML analysis results for later use

    for idx, filename in enumerate(input_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace('.JPG', '.jpg').replace('.PNG', '.png')
        output_path = os.path.join(output_dir, output_filename)

        if not os.path.exists(output_path):
            continue

        # Get file info
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / 1024

        # Analyze with ML detection (uses cached JSON if available)
        print(f"Processing {idx}/{total_images}: {filename}")
        ml_info = analyze_image_smart(input_path, output_path)

        # Cache ML analysis for JS data generation
        ml_analysis_cache[idx] = ml_info

        # Convert images to base64
        input_b64 = image_to_base64(input_path, max_height=600)
        output_b64 = image_to_base64(output_path, max_height=600)

        reduction = ((1 - (output_size/1024) / input_size) * 100)

        # Build detection overlays HTML
        detections_html = ""
        for det in ml_info['detections']:
            box_class = "detection-box primary" if det['is_primary'] else "detection-box"
            label_class = "detection-label primary" if det['is_primary'] else "detection-label"
            label_text = f"{'★ ' if det['is_primary'] else ''}{det['class_name']} {det['confidence']:.0%}"

            detections_html += f"""
                    <div class="{box_class}" style="left: {det['bbox_pct'][0]:.1f}%; top: {det['bbox_pct'][1]:.1f}%; width: {det['bbox_pct'][2]:.1f}%; height: {det['bbox_pct'][3]:.1f}%;">
                        <div class="{label_class}">{label_text}</div>
                    </div>"""

        # Build crop area overlay
        crop_html = f"""
                    <div class="crop-area" style="left: {ml_info['crop_box_pct'][0]:.1f}%; top: {ml_info['crop_box_pct'][1]:.1f}%; width: {ml_info['crop_box_pct'][2]:.1f}%; height: {ml_info['crop_box_pct'][3]:.1f}%;">
                        <div class="crop-label">SMART CROP AREA</div>
                    </div>"""

        # Build ML info summary
        ml_summary = ""
        if ml_info['detection_count'] > 0:
            primary_det = next((d for d in ml_info['detections'] if d['is_primary']), None)
            if primary_det:
                ml_summary = f"""<div class="ml-info">
                    <span class="badge primary">Primary: {primary_det['class_name']} ({primary_det['confidence']:.0%})</span>
                    {f'<span class="badge detected">+{ml_info["detection_count"]-1} more</span>' if ml_info['detection_count'] > 1 else ''}
                </div>"""
        else:
            ml_summary = '<div class="ml-info">No objects detected • Using center crop fallback</div>'

        html_content += f"""
    <div class="comparison" id="img-{idx}">
        <div class="comparison-header">
            <div class="filename">📷 {filename}</div>
            <div class="image-number">#{idx} of {total_images}</div>
        </div>
        <div class="images-container">
            <div class="image-box">
                <div class="image-label">Original ({ml_info['original']}) - {input_size:.2f} MB</div>
                <div class="image-wrapper">
                    <img src="{input_b64}" alt="Original">
                    {detections_html}
                    {crop_html}
                </div>
                {ml_summary}
            </div>
            <div class="image-box">
                <div class="image-label">Processed Output (480x800)</div>
                <div class="image-wrapper">
                    <img src="{output_b64}" alt="Processed">
                </div>
                <div class="stats">{output_size:.1f} KB • {reduction:.1f}% reduction</div>
            </div>
        </div>
        <div class="feedback-section">
            <div class="feedback-title">Rate Crop Quality:</div>
            <div class="rating-buttons">
                <button class="rating-btn excellent" onclick="setRating({idx}, 5, this)">
                    ⭐⭐⭐⭐⭐<br>Excellent
                </button>
                <button class="rating-btn good" onclick="setRating({idx}, 4, this)">
                    ⭐⭐⭐⭐<br>Good
                </button>
                <button class="rating-btn fair" onclick="setRating({idx}, 3, this)">
                    ⭐⭐⭐<br>Fair
                </button>
                <button class="rating-btn poor" onclick="setRating({idx}, 2, this)">
                    ⭐⭐<br>Poor
                </button>
                <button class="rating-btn bad" onclick="setRating({idx}, 1, this)">
                    ⭐<br>Bad
                </button>
            </div>
            <textarea class="comment-box" placeholder="Optional comments about this crop..."
                      id="comment-{idx}" onchange="saveComment({idx}, this.value)"></textarea>
        </div>
    </div>
"""

    # Build ML data JavaScript object from cached analysis
    ml_data_entries = []
    for idx, ml_info in ml_analysis_cache.items():
        # Format detections for JS
        detections_js = [
            f'{{"class_name": "{d["class_name"]}", "confidence": {d["confidence"]:.2f}, "is_primary": {str(d["is_primary"]).lower()}}}'
            for d in ml_info['detections']
        ]

        ml_data_entries.append(
            f'            {idx}: {{"detections": [{", ".join(detections_js)}], '
            f'"crop_box": {list(ml_info["crop_box"])}, "detection_count": {ml_info["detection_count"]}}}'
        )

    ml_data_js = ",\n".join(ml_data_entries)

    html_content += f"""
    <div class="export-section">
        <button class="export-btn" onclick="exportFeedback()">
            💾 Export Feedback
        </button>
        <div class="feedback-count" id="exportCount">0 rated</div>
    </div>

    <script>
        let feedback = {{}};
        const totalImages = {total_images};

        // Store ML analysis data for each image
        const mlAnalysisData = {{
{ml_data_js}
        }};

        function setRating(imageId, rating, button) {{
            // Remove selected from siblings
            const buttons = button.parentElement.querySelectorAll('.rating-btn');
            buttons.forEach(btn => btn.classList.remove('selected'));

            // Add selected to clicked button
            button.classList.add('selected');

            // Save rating with ML analysis data
            if (!feedback[imageId]) feedback[imageId] = {{}};
            feedback[imageId].rating = rating;
            feedback[imageId].filename = document.querySelector(`#img-${{imageId}} .filename`).textContent.replace('📷 ', '');

            // Include ML analysis data in feedback
            if (mlAnalysisData[imageId]) {{
                feedback[imageId].detections = mlAnalysisData[imageId].detections;
                feedback[imageId].crop_box = mlAnalysisData[imageId].crop_box;
                feedback[imageId].detection_count = mlAnalysisData[imageId].detection_count;
            }}

            updateStats();
            saveToLocalStorage();
        }}

        function saveComment(imageId, comment) {{
            if (!feedback[imageId]) feedback[imageId] = {{}};
            feedback[imageId].comment = comment;
            saveToLocalStorage();
        }}

        function updateStats() {{
            const ratedCount = Object.keys(feedback).filter(id => feedback[id].rating).length;
            const ratings = Object.values(feedback).filter(f => f.rating).map(f => f.rating);
            const avgRating = ratings.length > 0
                ? (ratings.reduce((a, b) => a + b, 0) / ratings.length).toFixed(1)
                : '-';

            document.getElementById('totalCount').textContent = totalImages;
            document.getElementById('ratedCount').textContent = ratedCount;
            document.getElementById('avgRating').textContent = avgRating;
            document.getElementById('exportCount').textContent = `${{ratedCount}} rated`;
        }}

        function saveToLocalStorage() {{
            localStorage.setItem('cropFeedback', JSON.stringify(feedback));
        }}

        function loadFromLocalStorage() {{
            const saved = localStorage.getItem('cropFeedback');
            if (saved) {{
                feedback = JSON.parse(saved);

                // Restore UI state
                Object.entries(feedback).forEach(([imageId, data]) => {{
                    if (data.rating) {{
                        const buttons = document.querySelectorAll(`#img-${{imageId}} .rating-btn`);
                        buttons[5 - data.rating].classList.add('selected');
                    }}
                    if (data.comment) {{
                        document.getElementById(`comment-${{imageId}}`).value = data.comment;
                    }}
                }});

                updateStats();
            }}
        }}

        function exportFeedback() {{
            const exportData = {{
                timestamp: new Date().toISOString(),
                totalImages: totalImages,
                ratedImages: Object.keys(feedback).filter(id => feedback[id].rating).length,
                feedback: feedback
            }};

            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `crop_quality_feedback_${{new Date().toISOString().split('T')[0]}}.json`;
            link.click();
            URL.revokeObjectURL(url);

            alert(`Exported feedback for ${{Object.keys(feedback).filter(id => feedback[id].rating).length}} images`);
        }}

        // Load saved feedback on page load
        window.addEventListener('DOMContentLoaded', () => {{
            loadFromLocalStorage();
            updateStats();
        }});
    </script>
</body>
</html>
"""

    # Save HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"\n✓ Quality report created: {html_path}")
    print(f"  Total images: {total_images}")
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML quality assessment report'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory with original images'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory with processed images'
    )
    parser.add_argument(
        '--html',
        default='quality_report.html',
        help='Path to save HTML report (default: quality_report.html)'
    )
    parser.add_argument(
        '--title',
        default='Crop Quality Assessment',
        help='Report title'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        return 1

    generate_html_report(
        args.input_dir,
        args.output_dir,
        args.html,
        args.title
    )

    print(f"\nOpen in browser: {args.html}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
