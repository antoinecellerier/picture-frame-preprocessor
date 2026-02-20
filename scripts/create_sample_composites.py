#!/usr/bin/env python3
"""Create before/after composite images for the README."""

from PIL import Image, ImageDraw, ImageOps
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "test_real_images" / "input"
OUTPUT = BASE / "test_real_images" / "output_optimized"
SAMPLES = BASE / "samples"
SAMPLES.mkdir(exist_ok=True)

ARROW_W = 40
PAD = 12
GAP = 28
BG = (24, 24, 24)


def load_image(path):
    img = Image.open(path)
    return ImageOps.exif_transpose(img)


def load_pair(name):
    for ext in (".JPG", ".jpg", ".jpeg", ".png"):
        inp = INPUT / f"{name}{ext}"
        if inp.exists():
            break
    out = OUTPUT / f"{name}.jpg"
    return load_image(inp), load_image(out)


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


if __name__ == "__main__":
    # Hero: 2 rows
    # Row 1: mural detection + sculpture zoom
    # Row 2: multi-crop street art
    multi_crop_outputs = [
        SAMPLES / "DSC_3089_crop_2.jpg",  # geisha
        SAMPLES / "DSC_3089_crop_1.jpg",  # girl in red
    ]
    create_hero_two_rows(
        row1_entries=["DSC_3614", "DSC_4168"],
        row2_entries=[("DSC_3089", multi_crop_outputs)],
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
