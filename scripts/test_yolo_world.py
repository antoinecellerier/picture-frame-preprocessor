#!/usr/bin/env python3
"""Test YOLO-World with art-specific prompts."""

from ultralytics import YOLOWorld
from PIL import Image
import sys

# Test on sample image
test_image = "test_real_images/input/DSC_1734.JPG"

print("Loading YOLO-World model...")
model = YOLOWorld('yolov8m-worldv2.pt')  # or yolov8l-worldv2

# Set art-specific classes
art_classes = [
    'sculpture', 'statue', 'painting', 'art installation',
    'mosaic', 'artwork', 'mural', 'art piece', 'exhibit'
]

model.set_classes(art_classes)

print(f"Testing on {test_image}")
print(f"Classes: {art_classes}")

# Run detection
results = model.predict(test_image, conf=0.25)

# Display results
for r in results:
    print(f"\nDetections found: {len(r.boxes)}")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = art_classes[cls_id] if cls_id < len(art_classes) else "unknown"
        bbox = box.xyxy[0].tolist()
        print(f"  - {cls_name}: {conf:.2f} at {bbox}")
