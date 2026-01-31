#!/usr/bin/env python3
"""Test improved prompts for better detection."""

import sys
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLOWorld

# Test different prompt strategies
prompt_sets = {
    'original': [
        'sculpture', 'statue', 'painting', 'art installation',
        'mosaic', 'artwork', 'mural', 'art piece',
        'exhibit', 'artistic object', 'wall art', 'decorative art'
    ],

    'specific_materials': [
        'stone sculpture', 'metal sculpture', 'wooden sculpture',
        'bronze statue', 'marble statue', 'ceramic sculpture',
        'oil painting', 'canvas painting', 'framed painting',
        'tile mosaic', 'glass mosaic', 'wall mosaic',
        'street art', 'graffiti art', 'mural painting'
    ],

    'contextual': [
        'museum exhibit', 'gallery display', 'art installation',
        'sculpture on pedestal', 'statue on display',
        'framed artwork', 'wall-mounted art',
        'decorative sculpture', 'artistic piece',
        'cultural artifact', 'historical sculpture'
    ],

    'expanded': [
        'sculpture', 'statue', 'figurine', 'bust',
        'painting', 'artwork', 'canvas', 'framed art',
        'mosaic', 'tile art', 'mural', 'wall art',
        'art installation', 'art piece', 'exhibit',
        'relief sculpture', 'monument', 'artistic object',
        'decorative art', 'cultural piece', 'pottery', 'vase'
    ]
}

# Test on a few sample images
test_images = [
    'test_real_images/input/DSC_1734.JPG',
    'test_real_images/input/DSC_2149.JPG',
    'test_real_images/input/DSC_2744.JPG'
]

print("Testing different prompt strategies on YOLO-World\n")

for prompt_name, prompts in prompt_sets.items():
    print(f"\n{'='*80}")
    print(f"Testing: {prompt_name.upper()}")
    print(f"Prompts: {', '.join(prompts[:5])}... ({len(prompts)} total)")
    print('='*80)

    model = YOLOWorld('yolov8m-worldv2.pt')
    model.set_classes(prompts)

    total_detections = 0

    for img_path in test_images:
        if not Path(img_path).exists():
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        results = model.predict(img, conf=0.25, verbose=False)

        det_count = sum(len(r.boxes) for r in results)
        total_detections += det_count

        print(f"  {Path(img_path).name}: {det_count} detections")

    avg = total_detections / len(test_images)
    print(f"  Average: {avg:.1f} detections per image")

print(f"\n{'='*80}")
print("RECOMMENDATION: Run full evaluation on best-performing prompt set")
print('='*80)
