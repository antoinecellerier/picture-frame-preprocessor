#!/usr/bin/env python3
"""Test Grounding DINO with art-specific text prompts."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T
import torch

def test_grounding_dino():
    """Test Grounding DINO on a sample image."""

    print("Loading Grounding DINO model...")

    # Art-specific text prompts
    text_prompt = "sculpture . painting . art installation . mosaic . statue . artwork . mural . exhibit"

    # Load model
    try:
        model = load_model(
            model_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="weights/groundingdino_swint_ogc.pth"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying to download model automatically...")

        # Try using the library's default model loading
        from groundingdino.util.inference import Model

        model = Model(
            model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="groundingdino_swint_ogc.pth"
        )

    # Test on sample image
    test_image_path = "test_real_images/input/20130317_020501_Android.jpg"

    if not Path(test_image_path).exists():
        print(f"Test image not found: {test_image_path}")
        return

    print(f"Testing on: {test_image_path}")
    print(f"Text prompt: {text_prompt}")

    # Load and process image
    image_source, image = load_image(test_image_path)

    # Run detection
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=0.3,
        text_threshold=0.25
    )

    print(f"\nDetections found: {len(boxes)}")
    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        print(f"  {i+1}. {phrase}: {logit:.2f}")
        print(f"     Box: {box.tolist()}")

if __name__ == '__main__':
    test_grounding_dino()
