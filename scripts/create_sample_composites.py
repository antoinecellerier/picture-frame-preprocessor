#!/usr/bin/env python3
"""Create before/after composite images and report screenshot for README.

Regenerates crop outputs using the current pipeline (not from stale files).
"""

import sys
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

INPUT = BASE / "test_real_images" / "input"
SAMPLES = BASE / "samples"
SAMPLES.mkdir(exist_ok=True)

ARROW_W = 40
PAD = 12
GAP = 28
BG = (24, 24, 24)

# Lazy-loaded pipeline components
_detector = None
_cropper = None


def _get_pipeline():
    global _detector, _cropper
    if _detector is None:
        from frame_prep.cli import create_detector, create_cropper
        from frame_prep import defaults
        # Use same defaults as CLI (VLM on, multi-crop on, etc.)
        _detector = create_detector(
            single_model=False, ensemble=False, model=None,
            confidence=None, no_two_pass=False, verbose=False)
        _cropper = create_cropper(
            width=defaults.TARGET_WIDTH, height=defaults.TARGET_HEIGHT,
            zoom=defaults.ZOOM_FACTOR)
    return _detector, _cropper


def load_image(path):
    img = Image.open(path)
    return ImageOps.exif_transpose(img)


def _find_input(name):
    for ext in (".JPG", ".jpg", ".jpeg", ".png"):
        p = INPUT / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No input found for {name}")


def generate_crop(name):
    """Generate a fresh crop using the current pipeline.

    Returns None if the image is filtered as non-art.
    """
    from frame_prep.pipeline import run_detection_pipeline
    from frame_prep import defaults

    detector, cropper = _get_pipeline()
    inp_path = _find_input(name)
    img = load_image(inp_path)
    result = run_detection_pipeline(img, detector, cropper, image_path=str(inp_path))
    if result.art_score < defaults.MIN_ART_SCORE:
        return None
    cropped = cropper.crop_image(img, result.filtered_detections,
                                  focal_detections=result.focal_detections)
    return cropped


def make_filtered_placeholder(width, height):
    """Create a dark placeholder image with 'FILTERED' text."""
    from PIL import ImageFont
    img = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    label = "FILTERED"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), label,
              fill=(255, 80, 80), font=font)
    # Subtitle
    try:
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        small = font
    sub = "non-art (text)"
    sb = draw.textbbox((0, 0), sub, font=small)
    sw = sb[2] - sb[0]
    draw.text(((width - sw) // 2, (height + th) // 2 + 10), sub,
              fill=(180, 180, 180), font=small)
    return img


def load_pair(name):
    inp_path = _find_input(name)
    inp_img = load_image(inp_path)
    out_img = generate_crop(name)
    if out_img is None:
        # Pipeline filtered this image — create placeholder
        # Match the target aspect ratio (480x800)
        out_img = make_filtered_placeholder(480, 800)
    return inp_img, out_img


def fit_height(img, h):
    ratio = h / img.height
    return img.resize((round(img.width * ratio), h), Image.LANCZOS)


def draw_arrow(draw, x, y, h, color=(200, 200, 200)):
    cy = y + h // 2
    draw.polygon([
        (x + ARROW_W - 14, cy),
        (x + 14, cy - 10),
        (x + 14, cy + 10),
    ], fill=color)


def load_pairs(entries):
    pair_data = []
    for entry in entries:
        if isinstance(entry, str):
            entry = (entry,)
        name = entry[0]
        extra_outputs = entry[1] if len(entry) > 1 else None
        inp_img, out_img = load_pair(name)
        out_imgs = [load_image(p) for p in extra_outputs] if extra_outputs else [out_img]
        pair_data.append((inp_img, out_imgs))
    return pair_data


def compute_width_for_height(pair_data, h):
    """Compute total width using aspect ratios only (no resize)."""
    w = PAD
    for inp_img, out_imgs in pair_data:
        w += round(inp_img.width / inp_img.height * h) + ARROW_W
        for i, out_img in enumerate(out_imgs):
            w += round(out_img.width / out_img.height * h)
            if i < len(out_imgs) - 1:
                w += 6
        w += GAP
    return w - GAP + PAD


def render(pair_data, pair_h, out_path):
    """Build and save the composite image with natural dimensions."""
    total_w = compute_width_for_height(pair_data, pair_h)
    total_h = pair_h + PAD * 2
    return render_centered(pair_data, pair_h, out_path, total_w, total_h)


def render_centered(pair_data, pair_h, out_path, canvas_w, canvas_h):
    """Build and save composite, centering content in canvas."""
    content_w = compute_width_for_height(pair_data, pair_h)
    content_h = pair_h + PAD * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    x_off = (canvas_w - content_w) // 2
    y_off = (canvas_h - content_h) // 2
    x = x_off + PAD

    for inp_img, out_imgs in pair_data:
        y = y_off + PAD
        inp_fit = fit_height(inp_img, pair_h)
        canvas.paste(inp_fit, (x, y))
        x += inp_fit.width
        draw_arrow(draw, x, y, pair_h)
        x += ARROW_W
        for i, out_img in enumerate(out_imgs):
            out_fit = fit_height(out_img, pair_h)
            canvas.paste(out_fit, (x, y))
            x += out_fit.width
            if i < len(out_imgs) - 1:
                x += 6
        x += GAP

    canvas.save(out_path, quality=88)
    ratio = canvas.width / canvas.height
    print(f"Saved {out_path} ({canvas.width}x{canvas.height}, {ratio:.2f}:1)")


def create_hero_two_rows(row1_entries, row2_entries, out_name, row_h=350):
    """Create hero composite with two rows of before/after pairs."""
    row1_data = load_pairs(row1_entries)
    row2_data = load_pairs(row2_entries)

    row1_w = compute_width_for_height(row1_data, row_h)
    row2_w = compute_width_for_height(row2_data, row_h)
    canvas_w = max(row1_w, row2_w)
    row_gap = 8
    canvas_h = (row_h + PAD * 2) * 2 + row_gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    def draw_row(pair_data, y_base):
        content_w = compute_width_for_height(pair_data, row_h)
        x = (canvas_w - content_w) // 2 + PAD
        y = y_base + PAD
        for inp_img, out_imgs in pair_data:
            inp_fit = fit_height(inp_img, row_h)
            canvas.paste(inp_fit, (x, y))
            x += inp_fit.width
            draw_arrow(draw, x, y, row_h)
            x += ARROW_W
            for i, out_img in enumerate(out_imgs):
                out_fit = fit_height(out_img, row_h)
                canvas.paste(out_fit, (x, y))
                x += out_fit.width
                if i < len(out_imgs) - 1:
                    x += 6
            x += GAP

    draw_row(row1_data, 0)
    draw_row(row2_data, row_h + PAD * 2 + row_gap)

    path = SAMPLES / out_name
    canvas.save(path, quality=88)
    ratio = canvas.width / canvas.height
    print(f"Saved {path} ({canvas.width}x{canvas.height}, {ratio:.2f}:1)")


def create_composite(entries, out_name, pair_h=450):
    """Create composite with natural aspect ratio."""
    pair_data = load_pairs(entries)
    render(pair_data, pair_h, SAMPLES / out_name)


def generate_multi_crops(name):
    """Generate multi-crop outputs using the current pipeline."""
    from frame_prep.pipeline import run_detection_pipeline

    detector, cropper = _get_pipeline()
    inp_path = _find_input(name)
    img = load_image(inp_path)
    result = run_detection_pipeline(img, detector, cropper, image_path=str(inp_path))
    crops = cropper.crop_all_subjects(img, result.filtered_detections,
                                       focal_detections=result.focal_detections)
    return [crop_img for crop_img, _, _ in crops]


def capture_report_screenshot(image_name=None):
    """Take a 1080p screenshot of the interactive report."""
    report_path = BASE / "reports" / "interactive_detection_report.html"
    if not report_path.exists():
        print("  Report not found — run 'frame-prep report' first")
        return
    print("Capturing report screenshot...")
    from playwright.sync_api import sync_playwright
    url = f"file://{report_path}"
    if image_name:
        url += f"#{image_name}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(url)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SAMPLES / "report_screenshot.png"))
        browser.close()
    print(f"  Saved: samples/report_screenshot.png")


if __name__ == "__main__":
    # Hero: 2 rows
    # Row 1: mural detection + sculpture zoom
    # Row 2: multi-crop street art
    multi_crops = generate_multi_crops("DSC_3089")
    # Save crops for the hero composite
    for i, crop in enumerate(multi_crops):
        crop.save(SAMPLES / f"DSC_3089_crop_{i+1}.jpg", quality=95)
    multi_crop_paths = [SAMPLES / f"DSC_3089_crop_{i+1}.jpg" for i in range(len(multi_crops))]

    create_hero_two_rows(
        row1_entries=["DSC_3614", "DSC_4168"],
        row2_entries=[("DSC_3089", multi_crop_paths)],
        out_name="hero_samples.jpg",
    )

    # Gallery: mosaic + sculpture
    create_composite(
        ["20220325_115329", "DSC_0771"],
        "sample_gallery.jpg",
    )

    # Street art
    create_composite(
        ["20220321_171136", "20220109_160326"],
        "sample_street_art.jpg",
    )

    # Focal point detection: wide mural → face/figure anchor crop
    create_composite(
        ["DSC_0153"],
        "sample_focal_detection.jpg",
    )

    # Text detection: text-heavy primary skipped / filtered
    # 20210424: graffiti text skipped → rat cook painting selected
    # DSC_3167: graffiti text "BONNE ANNÉE 2025" correctly filtered as non-art
    create_composite(
        ["20210424_155333", "DSC_3167"],
        "sample_text_detection.jpg",
    )

    # Report screenshot — show busy detection with off-center focal point
    capture_report_screenshot("20210529_153247.jpg")
