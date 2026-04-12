"""Complex sine conformal warp bot.

Fetches 3 images and applies the complex sine map as a backward warp:
  source = sin(z)

where z = frequency · (pixel - centre) / scale, normalised so the image
spans roughly [-π, π] in each axis at frequency=1.

Since sin(x + iy) = sin(x)·cosh(y) + i·cos(x)·sinh(y), the warp creates
periodic repetition horizontally and exponential stretch vertically —
producing interlocking wave/crescent patterns, especially with --tile.

Posts 3 warped images.
"""
import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from print_gallery_bot import _bilinear, _bilinear_wrap

logger = logging.getLogger(__name__)


def sine_warp(img_arr: np.ndarray, frequency: float = 1.0,
              tile: bool = False,
              cx: float = None, cy: float = None) -> np.ndarray:
    """Apply complex sine conformal warp (backward map: source = sin(z)).

    Pixel coordinates are normalised to a complex plane where the short side
    of the image spans [-π, π] at frequency=1. The backward map samples the
    source image at sin(z) for each output pixel z.

    Args:
        img_arr:   uint8 (H, W, 3) source image
        frequency: sine cycles across the image (default 1.0)
        tile:      tile source as Cartesian grid before warping
        cx, cy:    centre of warp (defaults to image centre)

    Returns:
        uint8 (H, W, 3) warped image
    """
    h, w = img_arr.shape[:2]
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    scale = min(w, h) / (2.0 * math.pi)

    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    zr = (xx - cx) / scale * frequency
    zi = (yy - cy) / scale * frequency

    # sin(x + iy) = sin(x)cosh(y) + i·cos(x)sinh(y)
    src_r = np.sin(zr) * np.cosh(zi)
    src_i = np.cos(zr) * np.sinh(zi)

    x_in = cx + src_r * scale / frequency
    y_in = cy + src_i * scale / frequency

    if tile:
        return _bilinear_wrap(img_arr, x_in, y_in)
    else:
        return _bilinear(img_arr, x_in, y_in)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Complex sine conformal warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./sine-warp-bot-output"))
    parser.add_argument("--frequency", type=float, default=1.0,
                        help="Sine cycles across the image (default 1.0)")
    parser.add_argument("--tile", action="store_true",
                        help="Tile source as Cartesian grid; warp bends the grid")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Warp params: frequency={args.frequency}, tile={args.tile}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, warping...")
        arr = np.array(img)
        warped = sine_warp(arr, frequency=args.frequency, tile=args.tile)
        dest = out_dir / f"sine_result_{i}.png"
        Image.fromarray(warped).save(dest)
        logger.info(f"  Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="sine-warp-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
