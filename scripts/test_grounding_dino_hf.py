#!/usr/bin/env python3
"""Test Grounding DINO using HuggingFace transformers (easier setup)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

def test_grounding_dino_hf():
    """Test Grounding DINO using HuggingFace API."""

    print("Loading Grounding DINO model from HuggingFace...")

    # Load model and processor
    model_id = "IDEA-Research/grounding-dino-tiny"

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        # Use CPU for now
        device = "cpu"
        model = model.to(device)

        print(f"✓ Model loaded on {device}")

        # Art-specific text prompts
        text_prompts = [
            "sculpture",
            "painting",
            "art installation",
            "mosaic",
            "statue",
            "artwork",
            "mural",
            "exhibit",
            "art piece",
            "artistic object"
        ]

        # Test on sample image
        test_image_path = "test_real_images/input/DSC_1734.JPG"

        if not Path(test_image_path).exists():
            # Try alternative
            test_image_path = "test_real_images/input/20130317_020501_Android.jpg"

        if not Path(test_image_path).exists():
            print("No test image found")
            return

        print(f"\nTesting on: {test_image_path}")
        print(f"Text prompts: {', '.join(text_prompts)}")

        # Load image
        image = Image.open(test_image_path)

        # Process inputs
        inputs = processor(
            images=image,
            text=text_prompts,
            return_tensors="pt"
        ).to(device)

        print("\nRunning inference...")

        # Run detection
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]

        # Display results
        print(f"\nDetections found: {len(results['scores'])}")

        for i, (score, label, box) in enumerate(zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        )):
            box = [int(x) for x in box.tolist()]
            print(f"  {i+1}. {label}: {score:.2f}")
            print(f"     Box: {box}")

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_grounding_dino_hf()
