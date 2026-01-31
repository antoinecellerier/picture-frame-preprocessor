#!/usr/bin/env python3
"""Cache YOLO-World with improved contextual prompts."""

import json
from pathlib import Path
from PIL import Image, ImageOps
import sys
from ultralytics import YOLOWorld
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

def cache_yolo_world_improved(ground_truth_path, confidence_threshold=0.25, cache_dir='cache'):
    """Run YOLO-World with improved prompts."""

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    input_dir = Path('test_real_images/input')

    print("Loading YOLO-World with improved prompts...")
    model = YOLOWorld('yolov8m-worldv2.pt')

    # Best-performing: Contextual + Expanded combination
    art_classes = [
        'museum exhibit', 'gallery display', 'art installation',
        'sculpture on pedestal', 'statue on display',
        'framed artwork', 'wall-mounted art',
        'decorative sculpture', 'artistic piece',
        'sculpture', 'statue', 'figurine', 'bust',
        'painting', 'artwork', 'canvas',
        'mosaic', 'tile art', 'mural', 'wall art',
        'relief sculpture', 'pottery', 'vase'
    ]

    model.set_classes(art_classes)

    print(f"✓ Model loaded with {len(art_classes)} improved classes")

    cache_file = cache_path / f"yolo_world_improved_conf{confidence_threshold}.json"
    all_detections = {}

    for gt_entry in tqdm(ground_truth, desc="Processing images"):
        filename = gt_entry['filename']
        img_path = input_dir / filename

        if not img_path.exists():
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        results = model.predict(img, conf=confidence_threshold, verbose=False)

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

    cache_data = {
        'model_name': 'yolo_world_improved',
        'confidence_threshold': confidence_threshold,
        'art_classes': art_classes,
        'detections': all_detections
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\n✓ Cached {len(all_detections)} images to: {cache_file}")

if __name__ == '__main__':
    cache_yolo_world_improved('test_real_images/ground_truth_annotations.json')
