"""Circle block letter bot.

Arranges the letters of a word around a circle, with each letter's 3D
extrusion pointing inward toward the center.  The result looks like the
letters are exploding outward from a central point.

Each letter is rendered using the same isometric block-letter pipeline as
block_letter_bot.py, then rotated so its extrusion aligns radially inward
and composited onto a square transparent canvas.

Posts 3 images.
"""
import argparse
import logging
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from block_letter_bot import (
    WORD_LIST,
    _available_fonts,
    _load_font,
    render_block_word,
)

logger = logging.getLogger(__name__)


def render_circle_word(word, src_arr, font_size=120, depth_px=60,
                       letter_spacing=12, max_letters=14, font_path=None):
    """Render letters of `word` in a circle, each extruding inward toward center.

    Letters are placed starting from 12 o'clock going clockwise, with spacing
    proportional to each letter's actual glyph width so they sit snugly around
    the ring.  The circle radius is computed so the circumference equals the
    total letter widths plus inter-letter spacing.

    Each letter's extrusion angle is computed from its position on the circle
    so that the 3D depth converges on the circle center — the vanishing point
    is at the center.

    Args:
        word:           Text to render (only printable non-space chars used)
        src_arr:        uint8 (H, W, 3) source image for texturing
        font_size:      Letter height in pixels (default 120)
        depth_px:       Extrusion depth in pixels (default 60)
        letter_spacing: Gap between letters in pixels (default 12)
        max_letters:    Maximum letters to place on the circle (default 14)
        font_path:      Optional explicit font path

    Returns:
        uint8 (H, W, 4) RGBA image, or None if word has no renderable chars
    """
    letters = [c for c in word.upper() if c.strip()][:max_letters]
    N = len(letters)
    if N == 0:
        return None

    # Measure each letter's advance width to compute a tight-fitting radius.
    font = _load_font(font_size, font_path)
    widths = []
    for char in letters:
        try:
            bb = font.getbbox(char)
            widths.append(max(1, bb[2] - bb[0]))
        except AttributeError:
            widths.append(font_size)

    total_arc = sum(widths) + N * letter_spacing
    circle_radius = max(1, int(total_arc / (2 * math.pi)))
    logger.debug(f"  letter widths={widths}, total_arc={total_arc}, radius={circle_radius}")

    # Cumulative arc-center position for each letter (in pixels along the arc).
    arc_centers = []
    cumulative = 0
    for w in widths:
        arc_centers.append(cumulative + w / 2)
        cumulative += w + letter_spacing

    # Canvas large enough to hold all rotated letters
    margin = font_size * 2 + depth_px * 2 + 80
    canvas_size = (circle_radius + margin) * 2
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    cx = cy = canvas_size // 2

    for i, char in enumerate(letters):
        # Arc angle from 12 o'clock going clockwise (radians).
        arc_angle_rad = arc_centers[i] / circle_radius
        # Convert to standard CCW angle from positive x-axis.
        theta = 90.0 - math.degrees(arc_angle_rad)

        # Compute the extrusion angle for this letter's position.
        # Derivation: after PIL CCW rotation by (210° - theta), an extrusion
        # rendered at angle alpha ends up in direction (cos(alpha + rot),
        # -sin(alpha + rot)) in screen coords.  Setting this equal to the
        # inward radial direction (-cos(theta), sin(theta)) requires
        # alpha + rot = 180° + theta, giving alpha = 2*theta - 30°.
        letter_angle_deg = 2.0 * theta - 30.0

        # Render single letter as RGBA using the block-letter pipeline.
        letter_arr = render_block_word(
            char, src_arr,
            font_size=font_size,
            depth_px=depth_px,
            angle_deg=letter_angle_deg,
            font_path=font_path,
        )
        letter_img = Image.fromarray(letter_arr)

        # Rotate so extrusion points toward center.
        # Derivation: standard extrusion direction in image coords is
        # (cos angle_deg, -sin angle_deg) ≈ upper-right.  PIL CCW rotation
        # by (210° - theta) maps that vector to (-cos theta, sin theta),
        # which is the inward radial direction for a letter at angle theta.
        pil_rotation = 210.0 - theta
        rotated = letter_img.rotate(
            pil_rotation, expand=True, resample=Image.BICUBIC,
        )

        # Place center of rotated letter image at its circle position.
        lx = int(cx + circle_radius * math.cos(math.radians(theta))) \
             - rotated.width // 2
        ly = int(cy - circle_radius * math.sin(math.radians(theta))) \
             - rotated.height // 2

        # Alpha-composite onto canvas.
        tmp = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        # Clamp paste position so it doesn't go out of bounds
        paste_x = max(0, lx)
        paste_y = max(0, ly)
        # Crop the rotated image if it would extend past canvas edge
        crop_x = paste_x - lx
        crop_y = paste_y - ly
        crop_w = min(rotated.width - crop_x, canvas_size - paste_x)
        crop_h = min(rotated.height - crop_y, canvas_size - paste_y)
        if crop_w > 0 and crop_h > 0:
            cropped = rotated.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            tmp.paste(cropped, (paste_x, paste_y))
        canvas = Image.alpha_composite(canvas, tmp)

    # Auto-crop to content with some padding
    arr = np.array(canvas)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if rows.any():
        pad = 40
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        r0 = max(0, r0 - pad)
        r1 = min(arr.shape[0], r1 + pad + 1)
        c0 = max(0, c0 - pad)
        c1 = min(arr.shape[1], c1 + pad + 1)
        arr = arr[r0:r1, c0:c1]
    return arr


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Circle block letter bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel",   default="img-junkyard")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("./circle-block-letter-bot-output"))
    parser.add_argument("--text-channel", default="song-titles",
                        help="Slack channel to fetch text from (default: song-titles)")
    parser.add_argument("--words", default="",
                        help="Comma-separated fallback word list")
    parser.add_argument("--font-size",     type=int,   default=120)
    parser.add_argument("--depth",         type=int,   default=60,
                        help="Extrusion depth in pixels (default 60)")
    parser.add_argument("--letter-spacing", type=int,   default=12,
                        help="Gap between letters in pixels (default 12)")
    parser.add_argument("--max-letters",   type=int,   default=14,
                        help="Max letters to place on the circle (default 14)")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images, fetch_random_message_texts
    from slack_poster import post_collages

    # Fetch word list from Slack text channel
    if args.text_channel:
        raw_titles = fetch_random_message_texts(token, args.text_channel, 50)
        word_list = []
        for t in raw_titles:
            for line in t.split("\n"):
                if re.search(r"https?://|www\.", line, re.IGNORECASE):
                    continue
                cleaned = re.sub(r"[^A-Za-z\s]", "", line).strip().upper()
                if cleaned and len(cleaned.split()) >= 1 \
                        and max(len(w) for w in cleaned.split()) <= 20:
                    word_list.append(cleaned)
        if not word_list:
            word_list = [w.strip().upper() for w in args.words.split(",") if w.strip()]
    else:
        word_list = [w.strip().upper() for w in args.words.split(",") if w.strip()]

    if not word_list:
        word_list = WORD_LIST

    source_dir = args.output_dir / "source"
    out_dir    = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick one font for this run (consistent across all 3 images)
    available_fonts = _available_fonts()
    if available_fonts:
        chosen_font_name, chosen_font_path = random.choice(available_fonts)
        logger.info(f"Using font: {chosen_font_name}")
    else:
        chosen_font_name, chosen_font_path = "default", None
        logger.warning("No catalog fonts found, using PIL default")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        src_h, src_w = arr.shape[:2]

        title = random.choice(word_list)
        # Use the first word of the title for the circle (short is better visually)
        word = title.split()[0] if title.split() else title
        logger.info(f"Image {i}: {src_w}×{src_h}, word='{word}'")

        result = render_circle_word(
            word, arr,
            font_size=args.font_size,
            depth_px=args.depth,
            letter_spacing=args.letter_spacing,
            max_letters=args.max_letters,
            font_path=chosen_font_path,
        )

        if result is None:
            logger.warning(f"  No renderable characters in '{word}', skipping")
            continue

        dest = out_dir / f"circle_block_result_{i}.png"
        Image.fromarray(result).save(dest)
        logger.info(f"  Saved {dest.name} ({result.shape[1]}×{result.shape[0]})")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="circle-block-letter-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
