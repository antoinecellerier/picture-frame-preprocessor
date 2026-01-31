#!/usr/bin/env python3
"""Cache Grounding DINO detections with Intel PyTorch Extension optimization."""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import sys
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import intel_extension_for_pytorch as ipex
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

def cache_grounding_dino_optimized(ground_truth_path, confidence_threshold=0.25, cache_dir='cache'):
    """Run Grounding DINO with Intel optimizations."""

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    input_dir = Path('test_real_images/input')

    print("Loading Grounding DINO model with Intel optimizations...")
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    device = "cpu"
    model = model.to(device)
    model.eval()

    # Apply Intel PyTorch Extension optimization
    print("Applying IPEX optimizations...")
    model = ipex.optimize(model)

    print(f"✓ Model loaded and optimized on {device}")

    # Art-specific text prompts
    text_prompts = [
        "sculpture",
        "statue",
        "painting",
        "art installation",
        "mosaic",
        "artwork",
        "mural",
        "art piece",
        "exhibit",
        "artistic object",
        "wall art",
        "decorative art"
    ]

    print(f"Using prompts: {', '.join(text_prompts)}")

    cache_file = cache_path / f"grounding_dino_optimized_conf{confidence_threshold}.json"

    all_detections = {}

    for idx, gt_entry in enumerate(tqdm(ground_truth, desc="Processing images")):
        filename = gt_entry['filename']

        img_path = input_dir / filename

        if not img_path.exists():
            print(f"  Warning: Image not found - {filename}")
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Process inputs
        inputs = processor(
            images=img,
            text=text_prompts,
            return_tensors="pt"
        ).to(device)

        # Run detection with torch.no_grad for efficiency
        with torch.no_grad(), torch.cpu.amp.autocast():  # Enable auto mixed precision
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence_threshold,
            target_sizes=[img.size[::-1]]  # (height, width)
        )[0]

        # Convert to our format
        detection_data = []
        for score, label, box in zip(
            results["scores"],
            results["text_labels"],
            results["boxes"]
        ):
            bbox = [int(x) for x in box.tolist()]
            detection_data.append({
                'bbox': bbox,
                'confidence': float(score),
                'class_name': label,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            })

        all_detections[filename] = {
            'detections': detection_data,
            'image_size': (img.width, img.height)
        }

    # Save cache
    cache_data = {
        'model_name': 'grounding_dino_optimized',
        'confidence_threshold': confidence_threshold,
        'text_prompts': text_prompts,
        'optimizations': ['ipex', 'amp'],
        'detections': all_detections
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\n✓ Cached {len(all_detections)} images to: {cache_file}")

def main():
    parser = argparse.ArgumentParser(description='Cache Grounding DINO detections (IPEX optimized)')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')

    args = parser.parse_args()

    cache_grounding_dino_optimized(
        args.ground_truth,
        confidence_threshold=args.confidence,
        cache_dir=args.cache_dir
    )

if __name__ == '__main__':
    main()
