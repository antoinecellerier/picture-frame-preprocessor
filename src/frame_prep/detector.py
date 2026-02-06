"""YOLO ML model wrapper for object detection (supports YOLOv8, YOLOv12, YOLO26)."""

import os
import json
import hashlib
import fcntl
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np
from pathlib import Path

# Optimize OpenVINO threading for CPU inference
# Use 8 threads per model instance for good balance
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENVINO_INFERENCE_NUM_THREADS', '8')

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
CACHE_DIR = PROJECT_ROOT / 'cache' / 'detections'


def _compute_image_hash(image: Image.Image) -> str:
    """Compute a hash of the image content for cache key generation."""
    img_bytes = image.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def _compute_path_hash(image_path: Union[str, Path]) -> str:
    """Compute a hash based on file path and modification time for quick cache lookup."""
    path = Path(image_path)
    if path.exists():
        stat = path.stat()
        key = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    return ""


def _compute_params_hash(*args) -> str:
    """Compute a hash of arbitrary parameters for cache key generation."""
    params_str = json.dumps(args, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]


def _get_cache_path(model_name: str, image_hash: str, params_hash: str) -> Path:
    """Get the cache file path for given model, image and parameters."""
    model_cache_dir = CACHE_DIR / model_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    return model_cache_dir / f"{image_hash}_{params_hash}.json"


def _load_cached_detections(cache_path: Path) -> Optional[List['Detection']]:
    """Load detections from cache file if it exists (with file locking for concurrent access)."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return [Detection(
            bbox=tuple(d['bbox']),
            confidence=d['confidence'],
            class_name=d['class_name'],
            area=d['area']
        ) for d in data]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _save_cached_detections(cache_path: Path, detections: List['Detection']) -> None:
    """Save detections to cache file (with file locking for concurrent access)."""
    data = [asdict(d) for d in detections]
    with open(cache_path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
        try:
            json.dump(data, f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclass
class Detection:
    """Object detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    area: int

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# ============================================================================
# Shared detection utilities (used by EnsembleDetector and OptimizedEnsembleDetector)
# ============================================================================

def calculate_iou(box1: Tuple[int, int, int, int],
                  box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def weighted_merge(detections: List[Detection]) -> Detection:
    """Merge detections using confidence-weighted average."""
    total_conf = sum(d.confidence for d in detections)

    # Weighted average of coordinates
    x1 = sum(d.bbox[0] * d.confidence for d in detections) / total_conf
    y1 = sum(d.bbox[1] * d.confidence for d in detections) / total_conf
    x2 = sum(d.bbox[2] * d.confidence for d in detections) / total_conf
    y2 = sum(d.bbox[3] * d.confidence for d in detections) / total_conf

    bbox = (int(x1), int(y1), int(x2), int(y2))

    # Use highest confidence and most common class name
    best_det = max(detections, key=lambda d: d.confidence)

    # Calculate merged area
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    return Detection(
        bbox=bbox,
        confidence=best_det.confidence,
        class_name=best_det.class_name,
        area=area
    )


def merge_boxes(detections: List[Detection], threshold: float) -> List[Detection]:
    """Merge overlapping detections using IoU threshold."""
    if not detections:
        return []

    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    merged = []
    used = set()

    for i, det1 in enumerate(detections):
        if i in used:
            continue

        # Find all detections that overlap with this one
        to_merge = [det1]

        for j, det2 in enumerate(detections[i+1:], start=i+1):
            if j in used:
                continue

            iou = calculate_iou(det1.bbox, det2.bbox)
            if iou > threshold:
                to_merge.append(det2)
                used.add(j)

        # Merge boxes (weighted average by confidence)
        if len(to_merge) == 1:
            merged.append(det1)
        else:
            merged_det = weighted_merge(to_merge)
            merged.append(merged_det)

    # Sort merged detections by confidence
    merged.sort(key=lambda d: d.confidence, reverse=True)

    return merged


class ArtFeatureDetector:
    """YOLO-based object detector for finding subjects in images."""

    def __init__(
        self,
        model_name: str = 'yolov8n',
        confidence_threshold: float = 0.25,
        use_openvino: bool = True
    ):
        """
        Initialize detector with specified model.

        Args:
            model_name: YOLO model variant (yolov8n/s/m, yolo12n/s/m, yolo26n/s/m, etc.)
            confidence_threshold: Minimum confidence for detections
            use_openvino: Use OpenVINO acceleration if available (default: True)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_openvino = use_openvino
        self._model = None
        self._model_type = None  # 'openvino' or 'pytorch'
        self._last_image_size = None  # Store for center-weighting

    def _load_model(self):
        """Lazy-load YOLO model on first use, preferring OpenVINO if available."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                # Ensure models directory exists
                MODELS_DIR.mkdir(parents=True, exist_ok=True)

                # Try OpenVINO model first if enabled
                if self.use_openvino:
                    # Check in models directory
                    openvino_path = MODELS_DIR / f'{self.model_name}_openvino_model'

                    if openvino_path.exists():
                        try:
                            self._model = YOLO(str(openvino_path), task='detect')
                            self._model_type = 'openvino'
                            return
                        except Exception as e:
                            # Fall back to PyTorch if OpenVINO fails
                            pass

                # Load PyTorch model from models directory
                # If it doesn't exist, Ultralytics will download it to this path
                model_path = MODELS_DIR / f'{self.model_name}.pt'
                self._model = YOLO(str(model_path))
                self._model_type = 'pytorch'

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YOLO model '{self.model_name}'. "
                    f"Run 'python scripts/download_models.py' to download models. "
                    f"Error: {e}"
                )

    def detect(self, image: Image.Image, verbose: bool = False) -> List[Detection]:
        """
        Run object detection on image.

        Args:
            image: PIL Image to process
            verbose: Print detection info

        Returns:
            List of Detection objects sorted by confidence (highest first)
        """
        self._load_model()

        # Store image size for center-weighting in get_primary_subject
        self._last_image_size = (image.width, image.height)

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Run inference
        results = self._model(img_array, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                # Handle cases where class_id is out of range (can happen with RT-DETR)
                class_name = result.names.get(cls_id, f'class_{cls_id}')
                area = (x2 - x1) * (y2 - y1)

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_name=class_name,
                    area=area
                ))

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        if verbose and detections:
            print(f"Found {len(detections)} detections:")
            for det in detections[:5]:  # Show top 5
                print(f"  - {det.class_name}: {det.confidence:.2f} at {det.bbox}")

        return detections

    def get_primary_subject(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Get the most likely primary subject from detections.

        Uses a scoring system that combines:
        - Class priority (art > other objects > people > avoid)
        - Strong center-weighting (objects in center are strongly preferred)
        - Size bonus (larger objects preferred, especially when centered)

        Art museums/galleries usually have people viewing the art, not as the subject.
        The art is typically centered in photos, while people are on the sides.
        """
        if not detections:
            return None

        # High-priority art classes from open-vocabulary models (YOLO-World, Grounding DINO)
        # These are the actual labels these models return for art
        high_priority_art_classes = {
            # Direct art terms
            'artwork', 'painting', 'mural', 'mosaic', 'fresco',
            'sculpture', 'statue', 'figurine', 'bust',
            'art installation', 'art piece', 'artistic object',
            'wall art', 'street art', 'graffiti',
            'exhibit', 'display', 'gallery piece',
            'decorative art', 'decorative piece',
            'relief', 'carving', 'engraving',
            'collage', 'mixed media',
            # Painted art (murals, figures on walls/signs)
            'painted figure', 'painted mural', 'painted art',
            # Compound labels from models
            'sculpture statue', 'art installation exhibit',
            'framed artwork', 'wall-mounted art',
            'museum exhibit', 'gallery display',
            'sculpture statue figurine',
        }

        # Classes that are likely to BE art or contain art (COCO classes)
        art_related_classes = {
            'vase', 'potted plant', 'clock', 'tv', 'laptop',  # Often decorative/art
            'kite',  # Often colorful art/patterns
            'bird', 'horse', 'elephant', 'bear', 'cat', 'dog',  # Animal statues/sculptures
            'boat', 'train', 'airplane',  # Vehicle sculptures/art
            'bench', 'chair', 'couch', 'dining table',  # Furniture art/installations
            'fire hydrant',  # Often painted/decorated as street art
            'umbrella',  # Colorful installations
            'bottle', 'cup', 'bowl',  # Glass/ceramic art
        }

        # Classes to strongly avoid (not art, distractions)
        # Note: 'sign' removed - street art often uses signs as medium
        avoid_classes = {
            # Trash/utility
            'trash', 'trash can', 'trash bin', 'garbage', 'garbage can',
            'recycling', 'recycling bin', 'dumpster', 'waste bin',
            # Structural/background
            'traffic light', 'parking meter',
            'window', 'door', 'wall', 'floor', 'ceiling',
            # Street furniture (not art)
            'street lamp', 'lamp post', 'light pole', 'street light',
            # Small incidental elements
            'signature', 'text', 'label',
            'light', 'lamp', 'outlet', 'switch',
        }

        # Get image center if available
        img_center_x = None
        img_center_y = None
        img_area = 1
        if self._last_image_size:
            img_width, img_height = self._last_image_size
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            img_area = img_width * img_height

        def get_class_multiplier(class_name: str) -> float:
            """Get priority multiplier based on class name."""
            class_lower = class_name.lower()

            # Check high-priority art classes (exact or substring match)
            # Only check if art_class is contained in class_lower (not reverse)
            # to avoid false positives like 'car' matching 'carving'
            for art_class in high_priority_art_classes:
                if art_class in class_lower:
                    return 4.0  # Highest priority for explicit art

            # Check avoid classes
            for avoid_class in avoid_classes:
                if avoid_class in class_lower:
                    return 0.05  # Strongly deprioritize

            # Check COCO art-related classes
            if class_lower in art_related_classes:
                return 2.5

            # Deprioritize people
            if class_lower == 'person':
                return 0.4

            # Default for unknown classes (could be art from open-vocab models)
            return 1.5

        def calculate_score(det: Detection) -> float:
            """
            Calculate priority score for a detection.

            Higher score = more likely to be the primary subject.
            """
            # Base score from confidence (0.0 - 1.0)
            score = det.confidence

            # Class priority multiplier
            class_multiplier = get_class_multiplier(det.class_name)
            score *= class_multiplier

            # Strong center-weighting: objects near center are strongly preferred
            if img_center_x and img_center_y:
                det_center_x, det_center_y = det.center

                # Calculate normalized distance from center (0.0 = center, 1.0 = corner)
                max_dist = ((img_center_x ** 2 + img_center_y ** 2) ** 0.5)
                actual_dist = (((det_center_x - img_center_x) ** 2 +
                               (det_center_y - img_center_y) ** 2) ** 0.5)
                normalized_dist = actual_dist / max_dist if max_dist > 0 else 0

                # Center bonus: 5.0x at center, 1.0x at corners (increased from 3.5x)
                # Using quadratic falloff for stronger center preference
                # This helps central art beat edge detections (e.g., street lamps)
                center_bonus = 1.0 + 4.0 * (1.0 - normalized_dist ** 2)
                score *= center_bonus

            # Size bonus (larger objects are more likely to be the subject)
            # More aggressive than before to prefer main subjects over small details
            size_ratio = det.area / img_area if img_area > 0 else 0

            # Progressive size bonus:
            # - Tiny (<1%): strong penalty - likely small detail, not main subject
            # - Very small (1-3%): moderate penalty
            # - Small (3-10%): neutral to slight bonus
            # - Medium (10-30%): good bonus
            # - Large (30-60%): best bonus - likely the main subject
            # - Too large (>60%): reduced bonus (likely background)
            if size_ratio < 0.01:
                size_bonus = 0.3  # Strong penalty for tiny objects
            elif size_ratio < 0.03:
                size_bonus = 0.6  # Moderate penalty
            elif size_ratio < 0.10:
                size_bonus = 1.0 + size_ratio * 3  # 1.0 to 1.3
            elif size_ratio < 0.30:
                size_bonus = 1.3 + size_ratio * 2  # 1.5 to 1.9
            elif size_ratio < 0.60:
                size_bonus = 2.0  # Strong bonus for large central subjects
            else:
                size_bonus = 1.3  # Reduced for very large (might be background)

            score *= size_bonus

            return score

        # Calculate scores and return highest
        scored_detections = [(det, calculate_score(det)) for det in detections]
        scored_detections.sort(key=lambda x: x[1], reverse=True)

        return scored_detections[0][0]


class EnsembleDetector:
    """
    Ensemble of multiple YOLO models with optimized box merging.

    Combines detections from multiple models and merges overlapping boxes
    to achieve higher accuracy (63.5% vs 38.1% for single model).

    Optimized configuration:
    - Models: YOLOv8m + RT-DETR-L
    - Merge threshold: 0.4
    - Achieves 63.5% image accuracy on art detection
    """

    def __init__(
        self,
        models: List[str] = None,
        confidence_threshold: float = 0.15,
        merge_threshold: float = 0.4,
        use_openvino: bool = True
    ):
        """
        Initialize ensemble detector.

        Args:
            models: List of model names (default: ['yolov8m', 'rtdetr-l'])
            confidence_threshold: Minimum confidence for detections
            merge_threshold: IoU threshold for merging overlapping boxes (0.4 optimal)
            use_openvino: Use OpenVINO acceleration if available
        """
        if models is None:
            models = ['yolov8m', 'rtdetr-l']

        self.models = models
        self.merge_threshold = merge_threshold
        self._detectors = []
        self._last_image_size = None

        # Initialize all detectors
        for model_name in models:
            detector = ArtFeatureDetector(
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                use_openvino=use_openvino
            )
            self._detectors.append(detector)


    def detect(self, image: Image.Image, verbose: bool = False) -> List[Detection]:
        """
        Run all detectors and merge overlapping boxes.

        Args:
            image: PIL Image to process
            verbose: Print detection info

        Returns:
            List of merged Detection objects sorted by confidence
        """
        self._last_image_size = (image.width, image.height)

        # Collect detections from all models
        all_detections = []

        for i, detector in enumerate(self._detectors):
            if verbose:
                print(f"Running {self.models[i]}...")

            detections = detector.detect(image, verbose=False)
            all_detections.extend(detections)

            if verbose:
                print(f"  Found {len(detections)} detections")

        if verbose:
            print(f"Total detections before merge: {len(all_detections)}")

        # Merge overlapping boxes
        merged_detections = merge_boxes(all_detections, self.merge_threshold)

        if verbose:
            print(f"Total detections after merge: {len(merged_detections)}")

        return merged_detections

    def get_primary_subject(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Get primary subject using center-weighting.

        Delegates to first detector's implementation.
        """
        if not detections:
            return None

        # Use first detector's get_primary_subject (it has center-weighting)
        return self._detectors[0].get_primary_subject(detections)


class OptimizedEnsembleDetector:
    """
    OPTIMIZED: YOLO-World + Grounding DINO ensemble for maximum accuracy.

    Achieves 96.8% accuracy (61/63 images) using:
    - YOLO-World with improved contextual prompts
    - Grounding DINO with art-specific prompts
    - Optimized merge threshold: 0.2

    This is the best-performing configuration for art detection.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        merge_threshold: float = 0.2,
        two_pass: bool = True
    ):
        """
        Initialize optimized ensemble detector.

        Args:
            confidence_threshold: Minimum confidence for detections (default: 0.25)
            merge_threshold: IoU threshold for merging (default: 0.2, optimized)
            two_pass: Enable two-pass center-crop detection (default: False).
                      Improves accuracy (~93% vs ~85%) but ~3-4x slower.
        """
        self.confidence_threshold = confidence_threshold
        self.merge_threshold = merge_threshold
        self.two_pass = two_pass
        self._last_image_size = None
        self._yolo_world = None
        self._grounding_dino = None
        self._dino_processor = None

        # YOLO-World class prompts (included in cache key)
        # Note: removed graffiti/stencil/bird statue - caused false positives
        self._art_classes = [
            # Museum/gallery art
            'museum exhibit', 'gallery display', 'art installation',
            'sculpture on pedestal', 'statue on display',
            'framed artwork', 'wall-mounted art',
            'decorative sculpture', 'artistic piece',
            # Sculptures and statues
            'sculpture', 'statue', 'figurine', 'bust',
            # Paintings and wall art
            'painting', 'artwork', 'canvas',
            'mosaic', 'tile art', 'mural', 'wall art',
            'relief sculpture', 'pottery', 'vase',
            # Street art specific (focused on painted art, not graffiti)
            'street art', 'painted figure', 'painted mural',
            'art on wall', 'decorated sign',
        ]

        # Grounding DINO prompts (included in cache key)
        self._dino_prompts = [
            'sculpture', 'statue', 'painting', 'art installation',
            'mosaic', 'artwork', 'mural', 'art piece',
            'exhibit', 'artistic object', 'wall art', 'decorative art',
            # Street art (no graffiti/stencil)
            'street art', 'painted figure', 'figurine',
        ]

    def _load_yolo_world(self):
        """Lazy-load YOLO-World model on first use."""
        if self._yolo_world is None:
            from ultralytics import YOLOWorld

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            yolo_world_path = MODELS_DIR / 'yolov8m-worldv2.pt'
            self._yolo_world = YOLOWorld(str(yolo_world_path))
            self._yolo_world.set_classes(self._art_classes)

    def _load_grounding_dino(self):
        """Lazy-load Grounding DINO model on first use."""
        if self._grounding_dino is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch

            model_id = "IDEA-Research/grounding-dino-tiny"
            try:
                self._dino_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=True)
                self._grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True)
            except OSError:
                self._dino_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
                self._grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self._grounding_dino = self._grounding_dino.to("cpu")
            self._grounding_dino.eval()

    def _run_yolo_world(self, image: Image.Image, image_hash: str, verbose: bool, path_hash: str = "") -> List[Detection]:
        """Run YOLO-World detection with caching."""
        # Cache key includes confidence threshold and class prompts
        params_hash = _compute_params_hash(self.confidence_threshold, self._art_classes)

        # Try path-based cache first (faster), then fall back to content hash
        cache_path = None
        if path_hash:
            cache_path = _get_cache_path("yolo_world", path_hash, params_hash)
            cached = _load_cached_detections(cache_path)
            if cached is not None:
                if verbose:
                    print(f"  YOLO-World: {len(cached)} detections (cached)")
                return cached

        cache_path = _get_cache_path("yolo_world", image_hash, params_hash)

        cached = _load_cached_detections(cache_path)
        if cached is not None:
            if verbose:
                print(f"  YOLO-World: {len(cached)} detections (cached)")
            return cached

        if verbose:
            print("Running YOLO-World...")

        self._load_yolo_world()
        yolo_results = self._yolo_world.predict(image, conf=self.confidence_threshold, verbose=False)

        detections = []
        for r in yolo_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                bbox = tuple(int(x) for x in box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                # Get actual class name from the model's names dict or our art_classes
                if cls_id < len(self._art_classes):
                    class_name = self._art_classes[cls_id]
                else:
                    class_name = r.names.get(cls_id, f"yolo:{cls_id}")

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_name=class_name,
                    area=area
                ))

        if verbose:
            print(f"  YOLO-World: {len(detections)} detections")

        # Save to content-based cache (and path-based if available)
        _save_cached_detections(cache_path, detections)
        if path_hash:
            path_cache = _get_cache_path("yolo_world", path_hash, params_hash)
            if path_cache != cache_path:
                _save_cached_detections(path_cache, detections)
        return detections

    def _run_grounding_dino(self, image: Image.Image, image_hash: str, verbose: bool, path_hash: str = "") -> List[Detection]:
        """Run Grounding DINO detection with caching."""
        # Cache key includes confidence threshold and prompts
        params_hash = _compute_params_hash(self.confidence_threshold, self._dino_prompts)

        # Try path-based cache first (faster), then fall back to content hash
        cache_path = None
        if path_hash:
            cache_path = _get_cache_path("grounding_dino", path_hash, params_hash)
            cached = _load_cached_detections(cache_path)
            if cached is not None:
                if verbose:
                    print(f"  Grounding DINO: {len(cached)} detections (cached)")
                return cached

        cache_path = _get_cache_path("grounding_dino", image_hash, params_hash)

        cached = _load_cached_detections(cache_path)
        if cached is not None:
            if verbose:
                print(f"  Grounding DINO: {len(cached)} detections (cached)")
            return cached

        if verbose:
            print("Running Grounding DINO...")

        self._load_grounding_dino()
        import torch

        inputs = self._dino_processor(
            images=image,
            text=self._dino_prompts,
            return_tensors="pt"
        ).to("cpu")

        with torch.no_grad():
            outputs = self._grounding_dino(**inputs)

        dino_results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        detections = []
        for score, label, box in zip(
            dino_results["scores"],
            dino_results["text_labels"],
            dino_results["boxes"]
        ):
            bbox = tuple(int(x) for x in box.tolist())
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            detections.append(Detection(
                bbox=bbox,
                confidence=float(score),
                class_name=label,
                area=area
            ))

        if verbose:
            print(f"  Grounding DINO: {len(detections)} detections")

        # Save to content-based cache (and path-based if available)
        _save_cached_detections(cache_path, detections)
        if path_hash:
            path_cache = _get_cache_path("grounding_dino", path_hash, params_hash)
            if path_cache != cache_path:
                _save_cached_detections(path_cache, detections)
        return detections

    def detect(self, image: Image.Image, verbose: bool = False, image_path: Union[str, Path, None] = None) -> List[Detection]:
        """
        Run both detectors and merge results with conditional two-pass detection.

        Uses a conditional two-pass approach:
        1. First pass: detect on full image with both models
        2. Second pass (only if needed): detect on center-cropped region (60%)
           - Skipped when pass 1 already found a viable art candidate in the
             central area, saving ~2x processing time for the majority of images
           - Triggered when no good central candidate exists, helping detect
             small or hard-to-find subjects

        Args:
            image: PIL Image to process
            verbose: Print detection info
            image_path: Optional path to image file for faster cache lookups

        Returns:
            List of merged Detection objects sorted by confidence
        """
        self._last_image_size = (image.width, image.height)

        # Compute hashes for caching
        path_hash = _compute_path_hash(image_path) if image_path else ""
        image_hash = _compute_image_hash(image)

        # === PASS 1: Full image detection ===
        yolo_detections = self._run_yolo_world(image, image_hash, verbose, path_hash)
        dino_detections = self._run_grounding_dino(image, image_hash, verbose, path_hash)

        all_detections = yolo_detections + dino_detections

        if verbose:
            print(f"  Pass 1 (full): {len(all_detections)} detections")

        # === PASS 2: Center-cropped detection (conditional) ===
        # Only run the expensive second pass if pass 1 didn't find viable
        # candidates in the central area of the image.
        if self.two_pass:
            if self._has_viable_central_candidate(all_detections, image.size):
                if verbose:
                    print("  Pass 2 skipped: viable central candidate found in pass 1")
            else:
                if verbose:
                    print("  Pass 2 triggered: no viable central candidate in pass 1")
                center_detections = self._detect_center_crop(image, image_hash, path_hash, verbose)
                all_detections.extend(center_detections)

        if verbose:
            print(f"  Total before merge: {len(all_detections)}")

        # Merge overlapping boxes (fast, no caching needed)
        merged_detections = merge_boxes(all_detections, self.merge_threshold)

        if verbose:
            print(f"  Total after merge: {len(merged_detections)}")

        return merged_detections

    def _has_viable_central_candidate(self, detections: List[Detection], image_size: tuple) -> bool:
        """
        Check if pass 1 found a viable art candidate in the central area.

        A candidate is viable if it's an art-related class (not person/avoid),
        has reasonable confidence, and its center falls within the middle 60%
        of the image (the same region pass 2 would crop).
        """
        if not detections:
            return False

        width, height = image_size
        img_area = width * height

        # Central region bounds (same 60% crop as pass 2)
        crop_ratio = 0.6
        left = width * (1 - crop_ratio) / 2
        top = height * (1 - crop_ratio) / 2
        right = width - left
        bottom = height - top

        # Classes that indicate a viable art subject
        art_keywords = {
            'artwork', 'painting', 'mural', 'mosaic', 'fresco',
            'sculpture', 'statue', 'figurine', 'bust',
            'art', 'exhibit', 'display', 'gallery',
            'vase', 'pottery', 'relief', 'carving',
        }

        # Classes to skip — not viable subjects
        skip_classes = {'person', 'trash', 'traffic light', 'parking meter'}

        for det in detections:
            cx, cy = det.center

            # Must be in central region
            if not (left <= cx <= right and top <= cy <= bottom):
                continue

            # Must not be a skip class
            class_lower = det.class_name.lower()
            if any(skip in class_lower for skip in skip_classes):
                continue

            # Must have reasonable confidence
            if det.confidence < 0.3:
                continue

            # Must not be tiny (< 1% of image) — too small to be a real subject
            size_ratio = det.area / img_area if img_area > 0 else 0
            if size_ratio < 0.01:
                continue

            # Check if it's art-related or at least not a known non-art class
            is_art = any(kw in class_lower for kw in art_keywords)
            is_large = size_ratio >= 0.05

            # Accept if it's art-related, or if it's a large central detection
            # (large central objects from open-vocab models are likely subjects)
            if is_art or is_large:
                return True

        return False

    def _detect_center_crop(self, image: Image.Image, image_hash: str, path_hash: str, verbose: bool) -> List[Detection]:
        """
        Run detection on center-cropped region and map coordinates back.

        Crops center 60% of image, runs detection, then maps bbox coordinates
        back to original image space.
        """
        width, height = image.size
        crop_ratio = 0.6  # Use center 60% of image

        # Calculate crop bounds
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop center region
        center_crop = image.crop((left, top, right, bottom))

        # Use modified hash for center crop cache
        center_hash = image_hash + "_center60"
        center_path_hash = path_hash + "_center60" if path_hash else ""

        # Run detection on cropped image
        yolo_dets = self._run_yolo_world(center_crop, center_hash, verbose=False, path_hash=center_path_hash)
        dino_dets = self._run_grounding_dino(center_crop, center_hash, verbose=False, path_hash=center_path_hash)

        # Map coordinates back to original image space
        center_detections = []
        for det in yolo_dets + dino_dets:
            x1, y1, x2, y2 = det.bbox
            mapped_bbox = (x1 + left, y1 + top, x2 + left, y2 + top)
            mapped_det = Detection(
                bbox=mapped_bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                area=(mapped_bbox[2] - mapped_bbox[0]) * (mapped_bbox[3] - mapped_bbox[1])
            )
            center_detections.append(mapped_det)

        if verbose:
            print(f"  Pass 2 (center): {len(center_detections)} detections")

        return center_detections

    def get_primary_subject(self, detections: List[Detection]) -> Optional[Detection]:
        """Get primary subject using center-weighting."""
        if not detections:
            return None

        # Use the first detector's get_primary_subject method
        # (we'll use the same center-weighting algorithm)
        detector = ArtFeatureDetector()
        detector._last_image_size = self._last_image_size
        return detector.get_primary_subject(detections)
