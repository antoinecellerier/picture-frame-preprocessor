#!/usr/bin/env python3
"""Cache YOLO-World detections for evaluation."""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import sys
from ultralytics import YOLOWorld
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

def cache_yolo_world(ground_truth_path, confidence_threshold=0.25, cache_dir='cache'):
    """Run YOLO-World and cache detections."""

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    input_dir = Path('test_real_images/input')

    print("Loading YOLO-World model...")
    model = YOLOWorld('yolov8m-worldv2.pt')

    # Art-specific text classes
    art_classes = [
        'sculpture', 'statue', 'painting', 'art installation',
        'mosaic', 'artwork', 'mural', 'art piece',
        'exhibit', 'artistic object', 'wall art', 'decorative art'
    ]

    model.set_classes(art_classes)

    print(f"✓ Model loaded")
    print(f"Using classes: {', '.join(art_classes)}")

    cache_file = cache_path / f"yolo_world_conf{confidence_threshold}.json"

    all_detections = {}

    for idx, gt_entry in enumerate(tqdm(ground_truth, desc="Processing images")):
        filename = gt_entry['filename']

        img_path = input_dir / filename

        if not img_path.exists():
            print(f"  Warning: Image not found - {filename}")
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Run detection
        results = model.predict(img, conf=confidence_threshold, verbose=False)

        # Convert to our format
        detection_data = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = [int(x) for x in box.xyxy[0].tolist()]
                cls_name = art_classes[cls_id] if cls_id < len(art_classes) else "unknown"

                detection_data.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class_name': cls_name,
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                })

        all_detections[filename] = {
            'detections': detection_data,
            'image_size': (img.width, img.height)
        }

    # Save cache
    cache_data = {
        'model_name': 'yolo_world',
        'confidence_threshold': confidence_threshold,
        'art_classes': art_classes,
        'detections': all_detections
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\n✓ Cached {len(all_detections)} images to: {cache_file}")

def main():
    parser = argparse.ArgumentParser(description='Cache YOLO-World detections')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')

    args = parser.parse_args()

    cache_yolo_world(
        args.ground_truth,
        confidence_threshold=args.confidence,
        cache_dir=args.cache_dir
    )

if __name__ == '__main__':
    main()
