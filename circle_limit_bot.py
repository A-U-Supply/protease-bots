"""Circle Limit conformal warp bot.

Fetches 3 images and applies the Poincaré disk backward warp — the
hyperbolic geometry underlying Escher's Circle Limit series (I–IV).

The backward map unrolls the hyperbolic plane into the unit disk:
  for each output pixel z (|z| < 1), sample source at w = 2·atanh(|z|/R) · z/|z|

where R is the disk radius. This maps the bounded disk to the infinite
hyperbolic plane. With --tile, the source image tiles that plane, producing
repeating copies that grow smaller and denser toward the disk boundary —
the characteristic Circle Limit structure.

With --polar, the source image is pre-warped into a circle using a polar
coordinate transform before tiling: columns wrap around the circumference,
rows map radially from centre to edge. This creates a meta effect —
circular tiles tiling inside a circle.

Pixels outside the outer disk are rendered black.

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


def polar_to_circle(img_arr: np.ndarray) -> np.ndarray:
    """Pre-warp a rectangular source image into a circular shape.

    Maps image columns around the 360° circumference (x → θ) and rows
    radially from the outer edge inward (y=0 → outer ring, y=h → centre).
    Output is a square image the size of the shorter side, black outside
    the circle.

    Args:
        img_arr: uint8 (H, W, 3) source image

    Returns:
        uint8 (S, S, 3) circular image where S = min(H, W)
    """
    h, w = img_arr.shape[:2]
    s = min(w, h)
    cx = s / 2.0
    cy = s / 2.0

    xs = np.arange(s, dtype=np.float64)
    ys = np.arange(s, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_norm = r / (s / 2.0)  # 0 at centre, 1 at edge

    theta = (np.arctan2(dy, dx) / (2.0 * math.pi)) % 1.0  # 0 to 1

    # Source x wraps around full circumference; source y maps outer→inner
    x_src = theta * w                # columns wrap around the circle
    y_src = r_norm * h               # rows go from centre (0) to outer (h)

    # Wrap x, clamp y (handled by _bilinear clipping)
    x_src_wrapped = x_src % w
    result = _bilinear(img_arr, x_src_wrapped, y_src)

    result[r_norm >= 1.0] = 0
    return result


def circle_limit_warp(img_arr: np.ndarray, radius: float = 0.95,
                      zoom: float = 6.0, tile: bool = False,
                      cx: float = None, cy: float = None) -> np.ndarray:
    """Apply Poincaré disk backward warp (Escher Circle Limit geometry).

    Normalises pixel coordinates to a unit disk of the given radius fraction.
    Applies the hyperbolic unrolling map w = 2·atanh(|z|/R)·(z/|z|) to get
    source coordinates. With tile=True the source image tiles the resulting
    infinite plane, producing the Circle Limit repetition pattern.

    Args:
        img_arr: uint8 (H, W, 3) source image
        radius:  fraction of image half-side that the disk occupies (default 0.95)
        zoom:    controls how many tile rings appear; higher = more rings (default 25.0)
        tile:    tile source as Cartesian grid (strongly recommended)
        cx, cy:  centre of the disk (defaults to image centre)

    Returns:
        uint8 (H, W, 3) warped image (black outside the disk)
    """
    h, w = img_arr.shape[:2]
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    norm = min(w, h) / 2.0

    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    zr = (xx - cx) / norm
    zi = (yy - cy) / norm
    r = np.sqrt(zr ** 2 + zi ** 2)

    outside_disk = r >= radius

    r_clamped = np.minimum(r, radius * (1.0 - 1e-7))
    r_safe = np.maximum(r_clamped, 1e-10)

    hyperbolic_r = 2.0 * np.arctanh(r_clamped / radius)

    source_scale = norm / (2.0 * math.pi) * zoom
    scale_factor = hyperbolic_r / r_safe * source_scale

    x_in = cx + scale_factor * zr
    y_in = cy + scale_factor * zi

    if tile:
        result = _bilinear_wrap(img_arr, x_in, y_in)
    else:
        result = _bilinear(img_arr, x_in, y_in)

    result[outside_disk] = 0
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Circle Limit (Poincaré disk) warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./circle-limit-bot-output"))
    parser.add_argument("--radius", type=float, default=0.95,
                        help="Disk radius as fraction of image half-side (default 0.95)")
    parser.add_argument("--zoom", type=float, default=6.0,
                        help="Tile ring density; higher = more rings visible (default 25.0)")
    parser.add_argument("--polar", action="store_true",
                        help="Pre-warp source into circle before tiling (meta effect)")
    parser.add_argument("--tile", action="store_true",
                        help="Tile source as repeating grid (recommended for Circle Limit effect)")
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

    logger.info(f"Warp params: radius={args.radius}, zoom={args.zoom}, "
                f"polar={args.polar}, tile={args.tile}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, warping...")
        arr = np.array(img)

        if args.polar:
            logger.info(f"  Applying polar pre-warp...")
            arr = polar_to_circle(arr)
            logger.info(f"  Polar output: {arr.shape[1]}×{arr.shape[0]}")

        warped = circle_limit_warp(arr, radius=args.radius, zoom=args.zoom, tile=args.tile)
        dest = out_dir / f"circle_limit_result_{i}.png"
        Image.fromarray(warped).save(dest)
        logger.info(f"  Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="circle-limit-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
