"""CLIP-based mosaic detector for semantic zero-shot classification."""

from typing import List, Optional
from pathlib import Path
from PIL import Image

from .detector import (
    Detection,
    CACHE_DIR,
    _compute_image_hash,
    _compute_params_hash,
    _get_cache_path,
)


class CLIPMosaicDetector:
    """Zero-shot mosaic detector using CLIP image-text similarity.

    Scores image crops against mosaic-positive and mosaic-negative prompts.
    Emits synthetic Detection(class_name='mosaic') objects for crops that
    exceed the threshold, letting the primary-subject scorer (5.0x multiplier
    for 'mosaic') pick them up naturally.
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    POSITIVE_PROMPTS = [
        "a photo of a mosaic made of small colorful tiles",
        "a decorative mosaic artwork on a wall",
        "tile mosaic art",
    ]
    NEGATIVE_PROMPTS = [
        "a photo that does not contain a mosaic",
    ]

    # Cache key: changes when prompts change
    _CACHE_PARAMS = (MODEL_ID, POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

    def __init__(self, threshold: float = 0.15, use_regions: bool = True):
        """
        Args:
            threshold: Minimum CLIP score to emit a mosaic Detection.
                       Score = mean(positive_sim) - mean(negative_sim).
                       Use 0.0 to score everything without emitting anything
                       (evaluation mode — check raw scores instead).
            use_regions: Also score each existing detection's bbox crop.
        """
        self.threshold = threshold
        self.use_regions = use_regions
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load_clip(self):
        """Lazy-load CLIP model on first use."""
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            try:
                self._clip_processor = CLIPProcessor.from_pretrained(
                    self.MODEL_ID, local_files_only=True
                )
                self._clip_model = CLIPModel.from_pretrained(
                    self.MODEL_ID, local_files_only=True
                )
            except OSError:
                self._clip_processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
                self._clip_model = CLIPModel.from_pretrained(self.MODEL_ID)

            self._clip_model.eval()

            # Pre-encode text prompts (done once, reused for every image).
            # Use internal text_model / text_projection to avoid transformers
            # version incompatibilities with get_text_features() return type.
            all_prompts = self.POSITIVE_PROMPTS + self.NEGATIVE_PROMPTS
            text_inputs = self._clip_processor(
                text=all_prompts, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                t_out = self._clip_model.text_model(**text_inputs)
                t_proj = self._clip_model.text_projection(t_out.pooler_output)
                t_proj = t_proj / t_proj.norm(dim=-1, keepdim=True)
            self._text_features = t_proj  # shape: (num_prompts, embed_dim)
            self._n_pos = len(self.POSITIVE_PROMPTS)

            # Store image processor inputs format for reuse
            self._img_proc = self._clip_processor

    # ------------------------------------------------------------------
    # Core scoring primitive
    # ------------------------------------------------------------------

    def score(self, image: Image.Image) -> float:
        """Return CLIP mosaic score for an image crop.

        Score = mean(positive_similarities) - mean(negative_similarities).
        Higher = more likely to be a mosaic.
        """
        import torch

        self._load_clip()

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_inputs = self._img_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            v_out = self._clip_model.vision_model(**img_inputs)
            v_proj = self._clip_model.visual_projection(v_out.pooler_output)
            v_proj = v_proj / v_proj.norm(dim=-1, keepdim=True)

        # Cosine similarities: shape (num_prompts,)
        sims = (v_proj @ self._text_features.T).squeeze(0)

        pos_mean = sims[: self._n_pos].mean().item()
        neg_mean = sims[self._n_pos :].mean().item()
        return pos_mean - neg_mean

    def _score_cached(self, image: Image.Image, region_tag: str) -> float:
        """Score with disk caching.  region_tag identifies the crop (e.g. 'full' or bbox string)."""
        img_hash = _compute_image_hash(image)
        params_hash = _compute_params_hash(self._CACHE_PARAMS, region_tag)
        cache_path = _get_cache_path("clip_mosaic", img_hash, params_hash)

        if cache_path.exists():
            try:
                import json

                return json.loads(cache_path.read_text())["score"]
            except Exception:
                pass

        s = self.score(image)

        try:
            import json

            cache_path.write_text(json.dumps({"score": s}))
        except Exception:
            pass

        return s

    # ------------------------------------------------------------------
    # Detection emitter
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Image.Image,
        existing_detections: List[Detection],
        candidate_detections: Optional[List[Detection]] = None,
        verbose: bool = False,
    ) -> List[Detection]:
        """Return new synthetic Detection(class_name='mosaic') objects.

        1. Scores each existing detection's bbox crop (if use_regions=True).
        2. Scores any additional candidate_detections bbox crops.
           These are typically low-confidence YOLO/DINO detections run at a
           looser threshold specifically to generate candidate zones for CLIP.
           They are NOT added to the main detection list — only their bboxes
           are used as regions to score.
        3. Scores the full image.
        Steps 1-3 all emit a Detection if score > self.threshold.

        Args:
            image: Full PIL image.
            existing_detections: Detections already found by YOLO/DINO.
            candidate_detections: Optional extra detections (e.g. low-confidence
                pass) used only as candidate regions for CLIP — never returned.
            verbose: Print per-crop scores.

        Returns:
            List of new synthetic Detection objects (may be empty).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        new_detections: List[Detection] = []

        # --- Per-region scoring (existing + candidate zones) ---
        if self.use_regions:
            all_candidates = list(existing_detections)
            if candidate_detections:
                all_candidates.extend(candidate_detections)
            # Deduplicate by bbox to avoid scoring identical crops twice
            seen_bboxes: set = set()
            deduped = []
            for det in all_candidates:
                key = det.bbox
                if key not in seen_bboxes:
                    seen_bboxes.add(key)
                    deduped.append(det)
            for det in deduped:
                x1, y1, x2, y2 = det.bbox
                # Clamp to image bounds
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(width, x2)
                y2c = min(height, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = image.crop((x1c, y1c, x2c, y2c))
                region_tag = f"{x1c},{y1c},{x2c},{y2c}"
                s = self._score_cached(crop, region_tag)
                if verbose:
                    print(
                        f"  CLIP region [{det.class_name} {det.bbox}]: {s:.4f}"
                        + (" ✓" if s > self.threshold else "")
                    )
                if s > self.threshold:
                    area = (x2c - x1c) * (y2c - y1c)
                    new_detections.append(
                        Detection(
                            bbox=(x1c, y1c, x2c, y2c),
                            confidence=float(s),
                            class_name="mosaic",
                            area=area,
                        )
                    )

        # --- Full-image scoring ---
        full_score = self._score_cached(image, "full")
        if verbose:
            print(
                f"  CLIP full image: {full_score:.4f}"
                + (" ✓" if full_score > self.threshold else "")
            )
        if full_score > self.threshold:
            new_detections.append(
                Detection(
                    bbox=(0, 0, width, height),
                    confidence=self._to_confidence(full_score),
                    class_name="mosaic",
                    area=width * height,
                )
            )

        return new_detections
