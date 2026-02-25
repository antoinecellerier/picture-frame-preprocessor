"""CLIP/SigLIP-based mosaic detector and class verifier for semantic zero-shot classification."""

from dataclasses import replace
from typing import List, Optional
from PIL import Image

from .detector import (
    Detection,
    ArtFeatureDetector,
    _compute_image_hash,
    _compute_params_hash,
    _get_cache_path,
)

# Shared SigLIP model cache: model_id → (model, img_proc, tokenizer)
# Both CLIPMosaicDetector (SigLIP variant) and SigLIPClassVerifier share loaded weights.
_SIGLIP_MODEL_CACHE: dict = {}


def _load_siglip_components(model_id: str):
    """Load SigLIP model components, returning (model, img_proc, tokenizer).

    Results are cached in _SIGLIP_MODEL_CACHE so the weights are only loaded once
    even when multiple detector instances are created.
    """
    if model_id in _SIGLIP_MODEL_CACHE:
        return _SIGLIP_MODEL_CACHE[model_id]

    from transformers import SiglipModel, SiglipImageProcessor, GemmaTokenizer

    lfo = {"local_files_only": True}
    try:
        img_proc = SiglipImageProcessor.from_pretrained(model_id, **lfo)
        tokenizer = GemmaTokenizer.from_pretrained(model_id, **lfo)
        model = SiglipModel.from_pretrained(model_id, **lfo)
    except OSError:
        img_proc = SiglipImageProcessor.from_pretrained(model_id)
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        model = SiglipModel.from_pretrained(model_id)

    model.eval()
    _SIGLIP_MODEL_CACHE[model_id] = (model, img_proc, tokenizer)
    return model, img_proc, tokenizer

# Default model IDs
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"


class CLIPMosaicDetector:
    """Zero-shot mosaic detector using CLIP or SigLIP2 image-text similarity.

    Scores image crops against mosaic-positive and mosaic-negative prompts.
    Emits synthetic Detection(class_name='mosaic') objects for crops that
    exceed the threshold, letting the primary-subject scorer (5.0x multiplier
    for 'mosaic') pick them up naturally.

    Supports two model families, selected via model_id:
      - CLIP  (openai/clip-vit-base-patch32): cosine similarity scores, ~±0.05 range
      - SigLIP2 (google/siglip2-base-patch16-224): sigmoid probabilities, cleaner 0-1 range
    """

    MODEL_ID = CLIP_MODEL_ID  # default; override via model_id param

    POSITIVE_PROMPTS = [
        "a photo of a mosaic made of small colorful tiles",
        "a decorative mosaic artwork on a wall",
        "tile mosaic art",
    ]
    NEGATIVE_PROMPTS = [
        "a photo that does not contain a mosaic",
    ]

    def __init__(
        self,
        threshold: float = 0.15,
        use_regions: bool = True,
        model_id: Optional[str] = None,
    ):
        """
        Args:
            threshold: Minimum score to emit a mosaic Detection.
                       CLIP:    score = mean(pos_cosine) - mean(neg_cosine), ~±0.05 range.
                       SigLIP2: score = mean(pos_prob)  - mean(neg_prob),   ~±0.5  range.
                       Use 0.0 to score everything without emitting (evaluation mode).
            use_regions: Score each existing detection's bbox crop in addition
                         to the full image.
            model_id: HuggingFace model ID. Defaults to CLIP_MODEL_ID.
                      Pass SIGLIP2_MODEL_ID to use SigLIP2 instead.
        """
        self.threshold = threshold
        self.use_regions = use_regions
        self.model_id = model_id or self.MODEL_ID
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------
    # Model family detection
    # ------------------------------------------------------------------

    def _is_siglip(self) -> bool:
        return "siglip" in self.model_id.lower()

    @property
    def _cache_params(self):
        """Cache key tuple — includes model_id so caches don't cross-contaminate."""
        return (self.model_id, self.POSITIVE_PROMPTS, self.NEGATIVE_PROMPTS)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_clip(self):
        """Lazy-load model on first use."""
        if self._clip_model is not None:
            return
        if self._is_siglip():
            self._load_siglip_model()
        else:
            self._load_clip_model()

    def _load_clip_model(self):
        """Load CLIP (openai/clip-vit-base-patch32 family)."""
        from transformers import CLIPModel, CLIPProcessor
        import torch

        try:
            self._clip_processor = CLIPProcessor.from_pretrained(
                self.model_id, local_files_only=True
            )
            self._clip_model = CLIPModel.from_pretrained(
                self.model_id, local_files_only=True
            )
        except OSError:
            self._clip_processor = CLIPProcessor.from_pretrained(self.model_id)
            self._clip_model = CLIPModel.from_pretrained(self.model_id)

        self._clip_model.eval()

        # Pre-encode text prompts once.
        # Use internal text_model/text_projection to avoid transformers >=5.0
        # API change where get_text_features() returns BaseModelOutputWithPooling.
        all_prompts = self.POSITIVE_PROMPTS + self.NEGATIVE_PROMPTS
        text_inputs = self._clip_processor(
            text=all_prompts, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            t_out = self._clip_model.text_model(**text_inputs)
            t_proj = self._clip_model.text_projection(t_out.pooler_output)
            t_proj = t_proj / t_proj.norm(dim=-1, keepdim=True)
        self._text_features = t_proj  # (num_prompts, embed_dim)
        self._n_pos = len(self.POSITIVE_PROMPTS)
        self._img_proc = self._clip_processor

    def _load_siglip_model(self):
        """Load SigLIP2 (google/siglip2-* family) using the shared component cache.

        SigLIP2 uses a GemmaTokenizer which is not registered in transformers
        5.0's auto-tokenizer table, breaking SiglipProcessor.from_pretrained().
        Work around by loading the image processor and tokenizer separately.
        """
        import torch

        model, img_proc, tokenizer = _load_siglip_components(self.model_id)
        self._clip_model = model
        self._img_proc = img_proc
        self._tokenizer = tokenizer

        # Pre-encode text prompts once (per-instance, since prompts differ).
        all_prompts = self.POSITIVE_PROMPTS + self.NEGATIVE_PROMPTS
        text_inputs = self._tokenizer(
            all_prompts, return_tensors="pt",
            padding="max_length", truncation=True, max_length=64,
        )
        with torch.no_grad():
            t_out = self._clip_model.text_model(**text_inputs)
            # SigLIP has no separate text_projection; pooler_output is the feature.
            t_feat = t_out.pooler_output
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        self._text_features = t_feat  # (num_prompts, embed_dim)
        self._n_pos = len(self.POSITIVE_PROMPTS)
        # _clip_processor alias used by CLIP path; not used for SigLIP scoring
        self._clip_processor = self._img_proc

        # Grab learned logit scale (and bias if present) for sigmoid scoring.
        self._siglip_logit_scale = self._clip_model.logit_scale.exp()
        self._siglip_logit_bias = getattr(self._clip_model, "logit_bias", None)

    # ------------------------------------------------------------------
    # Core scoring primitive
    # ------------------------------------------------------------------

    def score(self, image: Image.Image) -> float:
        """Return mosaic score for an image crop.

        CLIP:    score = mean(pos_cosine_sims) - mean(neg_cosine_sims)  (~±0.05)
        SigLIP2: score = mean(pos_sigmoid_probs) - mean(neg_sigmoid_probs) (~±0.5)
        Higher = more likely to be a mosaic.
        """
        import torch

        self._load_clip()

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self._is_siglip():
            return self._score_siglip(image)
        else:
            return self._score_clip(image)

    def _score_clip(self, image: Image.Image) -> float:
        """Score via CLIP cosine similarity."""
        import torch

        img_inputs = self._img_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            v_out = self._clip_model.vision_model(**img_inputs)
            v_proj = self._clip_model.visual_projection(v_out.pooler_output)
            v_proj = v_proj / v_proj.norm(dim=-1, keepdim=True)

        sims = (v_proj @ self._text_features.T).squeeze(0)
        pos_mean = sims[: self._n_pos].mean().item()
        neg_mean = sims[self._n_pos :].mean().item()
        return pos_mean - neg_mean

    def _score_siglip(self, image: Image.Image) -> float:
        """Score via SigLIP2 cosine similarity (same formula as CLIP).

        logit_scale/logit_bias are training artefacts — applying sigmoid with a
        large logit_scale saturates everything near 0/1 and destroys the
        fine-grained discrimination we need.  Plain cosine similarity between
        normalized embeddings works better for our difference-of-means score.
        """
        import torch

        img_inputs = self._img_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            v_out = self._clip_model.vision_model(**img_inputs)
            v_feat = v_out.pooler_output
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        sims = (v_feat @ self._text_features.T).squeeze(0)
        pos_mean = sims[: self._n_pos].mean().item()
        neg_mean = sims[self._n_pos :].mean().item()
        return pos_mean - neg_mean

    def _score_cached(self, image: Image.Image, region_tag: str) -> float:
        """Score with disk caching. region_tag identifies the crop (e.g. 'full' or bbox string)."""
        img_hash = _compute_image_hash(image)
        params_hash = _compute_params_hash(self._cache_params, region_tag)
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
                    confidence=float(full_score),
                    class_name="mosaic",
                    area=width * height,
                )
            )

        return new_detections


class SigLIPClassVerifier:
    """Verify and correct detection class labels using SigLIP2 image-text similarity.

    For each detection bbox crop, scores against artwork/person/scene semantic
    categories and corrects mislabels before primary subject selection:
      - Art-class detection (multiplier >= 2.0) + SigLIP says person
        → reclassify to "person"
      - Person-class detection + SigLIP says artwork
        → reclassify to "painted figure"
      - Either case + SigLIP says scene
        → soft-demote confidence × SCENE_CONFIDENCE_FACTOR (class unchanged)

    Reclassification only fires when the winner's margin over the runner-up
    exceeds RECLASSIFY_MARGIN to avoid flipping uncertain cases.
    """

    MODEL_ID = SIGLIP2_MODEL_ID

    CATEGORY_PROMPTS = {
        "artwork": [
            "a painting or drawing hanging on a wall",
            "a sculpture or statue displayed in a museum",
            "a piece of artwork or art object",
            "a framed picture, mosaic, or mural",
        ],
        "person": [
            "a real person standing or walking in a gallery",
            "a museum visitor or human being photographed",
            "a person's body or face in the scene",
        ],
        "scene": [
            "an empty wall, floor, or room interior",
            "architectural detail or building facade",
            "furniture, signage, or non-art object",
        ],
    }

    # Class-name keywords that identify unambiguously art objects.
    # Detections whose class_name contains any of these keywords are never
    # reclassified to "person" — a mosaic or a framed painting cannot be a
    # real museum visitor no matter what SigLIP says.
    _CLEARLY_ART_KEYWORDS = frozenset({
        'mosaic', 'mural', 'fresco', 'painting', 'framed',
        'wall art', 'wall-mounted', 'street art', 'graffiti',
        'vase', 'pottery', 'ceramic', 'tapestry', 'sculpture',
        'art installation', 'figurine', 'artistic',
    })

    RECLASSIFY_MARGIN = 0.003
    SCENE_CONFIDENCE_FACTOR = 0.4

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or self.MODEL_ID
        self._model = None
        self._img_proc = None
        self._tokenizer = None
        self._text_features = None  # Tensor (N_total_prompts, embed_dim)
        self._category_slices: dict = {}  # cat → (start_idx, end_idx)

    def _load(self):
        """Load model (via shared cache) and pre-encode category text prompts."""
        if self._model is not None:
            return
        import torch

        model, img_proc, tokenizer = _load_siglip_components(self.model_id)
        self._model = model
        self._img_proc = img_proc
        self._tokenizer = tokenizer

        # Concatenate all category prompts and record slice positions
        all_prompts = []
        slices = {}
        start = 0
        for cat, prompts in self.CATEGORY_PROMPTS.items():
            slices[cat] = (start, start + len(prompts))
            all_prompts.extend(prompts)
            start += len(prompts)
        self._category_slices = slices

        text_inputs = self._tokenizer(
            all_prompts, return_tensors="pt",
            padding="max_length", truncation=True, max_length=64,
        )
        with torch.no_grad():
            t_out = self._model.text_model(**text_inputs)
            t_feat = t_out.pooler_output
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        self._text_features = t_feat  # (N_prompts, embed_dim)

    def classify_crop(self, crop: Image.Image) -> tuple:
        """Return (winner_category, winner_score, all_scores_dict, margin).

        Scores the crop against each category via mean cosine similarity of
        normalized SigLIP embeddings.
        """
        import torch

        self._load()

        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        img_inputs = self._img_proc(images=crop, return_tensors="pt")
        with torch.no_grad():
            v_out = self._model.vision_model(**img_inputs)
            v_feat = v_out.pooler_output
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        sims = (v_feat @ self._text_features.T).squeeze(0)  # (N_prompts,)

        scores = {}
        for cat, (i, j) in self._category_slices.items():
            scores[cat] = sims[i:j].mean().item()

        winner = max(scores, key=lambda k: scores[k])
        winner_score = scores[winner]
        sorted_vals = sorted(scores.values(), reverse=True)
        margin = winner_score - (sorted_vals[1] if len(sorted_vals) > 1 else winner_score)

        return winner, winner_score, scores, margin

    def verify(self, detection: Detection, image: Image.Image,
               img_w: int, img_h: int, verbose: bool = False) -> Detection:
        """Verify one detection; return corrected Detection (or original if unchanged).

        Creates a new Detection instance via dataclasses.replace — never mutates
        the original, keeping the pipeline safe for repeated calls.
        """
        x1, y1, x2, y2 = detection.bbox
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(img_w, x2), min(img_h, y2)

        if x2c <= x1c or y2c <= y1c:
            return detection

        crop = image.crop((x1c, y1c, x2c, y2c))
        winner, winner_score, scores, margin = self.classify_crop(crop)

        if verbose:
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(
                f"  SigLIP [{detection.class_name}]: {scores_str}"
                f" → {winner} (margin={margin:.4f})"
            )

        if margin < self.RECLASSIFY_MARGIN:
            return detection  # Too uncertain — don't reclassify

        multiplier = ArtFeatureDetector._get_class_multiplier(detection.class_name)
        original_class = detection.class_name
        class_lower = original_class.lower()

        # Art-class → person reclassification.
        # Guard: skip class names that are unambiguously artworks (a mosaic or
        # a framed painting cannot be a real person regardless of SigLIP score).
        if multiplier >= 2.0 and winner == "person":
            if any(kw in class_lower for kw in self._CLEARLY_ART_KEYWORDS):
                if verbose:
                    print(f"    → Skipped (clearly-art class): {original_class}")
            else:
                if verbose:
                    print(f"    → Reclassified: {original_class} → person")
                return replace(detection, class_name="person", original_class=original_class)

        # Person → artwork reclassification
        if detection.class_name.lower() == "person" and winner == "artwork":
            if verbose:
                print(f"    → Reclassified: person → painted figure")
            return replace(detection, class_name="painted figure", original_class=original_class)

        # Scene demotion (confidence penalty, class preserved).
        # Skip for clearly-art classes — a framed artwork or mosaic scored as
        # "scene" is almost always a false signal from background context.
        if winner == "scene":
            if any(kw in class_lower for kw in self._CLEARLY_ART_KEYWORDS):
                if verbose:
                    print(f"    → Skipped scene demote (clearly-art class): {original_class}")
            else:
                new_conf = detection.confidence * self.SCENE_CONFIDENCE_FACTOR
                if verbose:
                    print(
                        f"    → Scene demote: {original_class} conf"
                        f" {detection.confidence:.3f} → {new_conf:.3f}"
                    )
                return replace(detection, confidence=new_conf, original_class=original_class)

        return detection

    def verify_all(self, detections: List[Detection], image: Image.Image,
                   img_w: int, img_h: int, verbose: bool = False) -> List[Detection]:
        """Verify every detection; return new list with corrections applied."""
        return [
            self.verify(d, image, img_w, img_h, verbose=verbose)
            for d in detections
        ]
