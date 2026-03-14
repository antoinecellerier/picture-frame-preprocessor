"""Default configuration values shared across CLI, scripts, and reports.

This is the single source of truth for all defaults. CLI flags, batch
workers, sample scripts, and report generation all reference these.
"""

# Target frame dimensions
TARGET_WIDTH = 480
TARGET_HEIGHT = 800

# Detection
CONFIDENCE_THRESHOLD = 0.25
MERGE_THRESHOLD = 0.2
TWO_PASS = True

# VLM
USE_VLM = True
VLM_CONFIRM = False
VLM_MAX_IMAGE_SIZE = 512

# Multi-crop
MULTI_CROP = True

# Cropping
ZOOM_FACTOR = 8.0
USE_SALIENCY_FALLBACK = True
STRATEGY = 'smart'

# Non-art filtering
FILTER_NON_ART = True
MIN_ART_SCORE = 0.5  # confidence * class_multiplier; catches non-art (0.08) while keeping real art (>=0.55)

# Output
JPEG_QUALITY = 95
