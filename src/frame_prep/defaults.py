"""Default configuration values shared across CLI, scripts, and reports."""

# Target frame dimensions
TARGET_WIDTH = 480
TARGET_HEIGHT = 800

# Detection
CONFIDENCE_THRESHOLD = 0.25
MERGE_THRESHOLD = 0.2
TWO_PASS = True

# Cropping
ZOOM_FACTOR = 1.3
USE_SALIENCY_FALLBACK = True
STRATEGY = 'smart'

# Output
JPEG_QUALITY = 95
