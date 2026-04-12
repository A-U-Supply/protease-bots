"""3D block letter texture bot.

Fetches 3 source images and maps each as a texture onto isometric 3D block
letters spelling a word chosen randomly from a list.

For each letter:
  - Front face: source image texture, masked to the letter glyph outline
  - Top face:   source image texture, dimmed (simulates top lighting)
  - Right face: source image texture, darker (simulates shadow side)

UV coordinates are computed per-face using parallelogram inversion so the
texture wraps correctly onto each angled surface.

Posts 3 images.
"""
import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

WORD_LIST = [
    "PULSE", "WAVE", "FLUX", "ECHO", "LOOP",
    "VOID", "APEX", "NEON", "GRID", "HAZE",
    "BEAM", "CORE", "EDGE", "FLOW", "GLOW",
    "HEAT", "IRIS", "KNOT", "LENS", "NOVA",
    "PRISM", "SHIFT", "SPARK", "TRACE", "VEIL",
]

# Try these font paths in order (covers Ubuntu CI + macOS dev)
_FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    logger.warning("No truetype font found, falling back to default")
    return ImageFont.load_default()


def _face_uv(xx, yy, origin, vec_a, vec_b):
    """Compute (u, v) for every pixel relative to parallelogram face.

    Face is: origin + u·vec_a + v·vec_b,  u,v ∈ [0, 1].
    Returns u, v (full-image arrays) and boolean inside mask.
    """
    det = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
    if abs(det) < 1e-6:
        return np.zeros_like(xx), np.zeros_like(yy), np.zeros(xx.shape, bool)
    dx = xx - origin[0]
    dy = yy - origin[1]
    u = (dx * vec_b[1] - dy * vec_b[0]) / det
    v = (dy * vec_a[0] - dx * vec_a[1]) / det
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return u, v, inside


def _paint_face(output, src_f, xx, yy, origin, vec_a, vec_b, brightness,
                glyph_mask=None, glyph_W=None, glyph_H=None):
    """Sample source image onto a parallelogram face and write into output."""
    u, v, inside = _face_uv(xx, yy, origin, vec_a, vec_b)
    if not inside.any():
        return

    if glyph_mask is not None:
        # Mask front face to letter glyph outline
        u_px = (u * (glyph_W - 1)).clip(0, glyph_W - 1).astype(np.int32)
        v_px = (v * (glyph_H - 1)).clip(0, glyph_H - 1).astype(np.int32)
        inside = inside & glyph_mask[v_px, u_px]
        if not inside.any():
            return

    sh, sw = src_f.shape[:2]
    sx = (u[inside] * (sw - 1)).clip(0, sw - 1).astype(np.int32)
    sy = (v[inside] * (sh - 1)).clip(0, sh - 1).astype(np.int32)
    output[inside] = (src_f[sy, sx] * brightness).clip(0, 255)


def render_block_word(word, src_arr, font_size=200, depth_px=70, angle_deg=30,
                      letter_spacing=12, shade_top=0.78, shade_right=0.52):
    """Render the source image as isometric 3D block letter texture.

    Args:
        word:           text to render
        src_arr:        uint8 (H, W, 3) source image
        font_size:      letter height in pixels (default 200)
        depth_px:       extrusion depth (default 70)
        angle_deg:      isometric angle in degrees (default 30)
        letter_spacing: pixels between letters (default 12)
        shade_top:      top face brightness (default 0.78)
        shade_right:    right face brightness (default 0.52)

    Returns:
        uint8 (out_H, out_W, 3) rendered image
    """
    font = _load_font(font_size)

    rad = math.radians(angle_deg)
    iso_dx = int(depth_px * math.cos(rad))   # rightward component of extrusion
    iso_dy = int(depth_px * math.sin(rad))   # upward component (positive = up)

    # Measure characters
    chars = list(word)
    char_data = []
    for c in chars:
        bbox = font.getbbox(c)           # (left, top, right, bottom)
        W = bbox[2] - bbox[0]
        H = bbox[3] - bbox[1]
        char_data.append((c, bbox, W, H))

    total_w = sum(d[2] for d in char_data) + letter_spacing * max(0, len(chars) - 1)
    max_h   = max(d[3] for d in char_data)

    pad    = 40
    out_w  = total_w + iso_dx + 2 * pad
    out_h  = max_h   + iso_dy + 2 * pad

    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    src_f  = src_arr.astype(np.float32)

    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    x_cursor = iso_dx + pad     # leave room on left for extrusion overhang
    y0_base  = iso_dy + pad     # top-left of each front face

    vec_back = np.array([iso_dx, -iso_dy], dtype=float)  # extrusion direction

    # Collect faces; paint right→top→front so front is on top
    right_faces = []
    top_faces   = []
    front_faces = []

    for c, bbox, W, H in char_data:
        x0 = x_cursor
        y0 = y0_base

        vec_right = np.array([W, 0.0])
        vec_down  = np.array([0.0, H])

        # Glyph mask for front face
        glyph_img = Image.new('L', (W, H), 0)
        ImageDraw.Draw(glyph_img).text((-bbox[0], -bbox[1]), c, font=font, fill=255)
        glyph_mask = np.array(glyph_img) > 128

        # Right face: top-right corner → back → down
        right_faces.append(((x0 + W, y0), vec_back, vec_down, None, None, None))
        # Top face: top-left corner → right → back
        top_faces.append(((x0, y0), vec_right, vec_back, None, None, None))
        # Front face: top-left corner → right → down, masked to glyph
        front_faces.append(((x0, y0), vec_right, vec_down, glyph_mask, W, H))

        x_cursor += W + letter_spacing

    for origin, va, vb, gm, gW, gH in right_faces:
        _paint_face(output, src_f, xx, yy, origin, va, vb, shade_right, gm, gW, gH)
    for origin, va, vb, gm, gW, gH in top_faces:
        _paint_face(output, src_f, xx, yy, origin, va, vb, shade_top, gm, gW, gH)
    for origin, va, vb, gm, gW, gH in front_faces:
        _paint_face(output, src_f, xx, yy, origin, va, vb, 1.0, gm, gW, gH)

    return output.clip(0, 255).astype(np.uint8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="3D block letter texture bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel",   default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./block-letter-bot-output"))
    parser.add_argument("--words", default=",".join(WORD_LIST),
                        help="Comma-separated word list to choose from")
    parser.add_argument("--font-size", type=int, default=200)
    parser.add_argument("--depth",     type=int, default=70,
                        help="Extrusion depth in pixels (default 70)")
    parser.add_argument("--angle",     type=float, default=30.0,
                        help="Isometric angle in degrees (default 30)")
    parser.add_argument("--no-post",   action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages

    word_list = [w.strip().upper() for w in args.words.split(",") if w.strip()]

    source_dir = args.output_dir / "source"
    out_dir    = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img  = Image.open(path).convert("RGB")
        word = random.choice(word_list)
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, word='{word}', rendering...")
        arr    = np.array(img)
        result = render_block_word(word, arr, font_size=args.font_size,
                                   depth_px=args.depth, angle_deg=args.angle)
        dest = out_dir / f"block_letter_result_{i}.png"
        Image.fromarray(result).save(dest)
        logger.info(f"  Saved {dest.name} ({result.shape[1]}×{result.shape[0]})")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="block-letter-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
