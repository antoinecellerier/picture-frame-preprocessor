# Backlog

## Detection Quality

### P0: Boost art-class priority in primary selection
The #1 failure mode (29/39 bad). Mosaic, tile art, painted figure, sculpture, figurine detections are found by the detector but lose primary selection to generic COCO classes (person, bench, traffic light, etc.). Art-specific classes need a significant scoring bonus.

**Sub-issues:**
- Mosaic/tile art not selected as primary (18 images)
- Painted figure/figurine losing to other classes (11 images)
- Sculpture/statue/art installation not winning (6 images)

### P1: Sign vs art disambiguation
Street name signs occasionally beat nearby art in primary selection. "sign" was removed from `avoid_classes` to support street art/decorated signs, but regular signs now win. Need smarter heuristic (e.g., "decorated sign" vs plain "sign").

### P2: Incomplete mural detection
Models only bbox a small portion of large murals. Full mural coverage would improve crops. May need post-processing to merge adjacent mural detections.

### P2: Investigate zoom bug (#100)
Detection is reported correct but the crop/zoom is applied elsewhere. May be a report visualization issue or a cropper bug.

## Filtering

### P1: Non-art image filtering
Auto-detect and skip images that aren't interesting art subjects:
- Museum exhibit labels / info cards
- Photos of cars, buildings without art
- Images dominated by glass reflections (photographer's reflection visible)

6 images already marked `"not_art": true` in ground truth. Pipeline should skip these.

## Feature Requests

### P2: Smart sub-crop for wide subjects
When the detected art bbox is very wide (e.g., a panoramic mural), zoom into the most interesting detail area (face, focal point) rather than showing the full wide bbox at low zoom.

### P3: Multi-crop for panoramic scenes
Generate multiple output crops from a single image when it contains multiple distinct art pieces (e.g., 3-column mural wall). Currently only one primary is selected.
