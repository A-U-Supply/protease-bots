"""Escher/Lenstra log-spiral conformal warp bot.

Fetches 3 images and applies the Escher/Lenstra conformal map to each.

The map is the complex power h(w) = w^α where:
  α = (2πi + ln(scale)) / (2πi)  — de Smit & Lenstra, 2003

In polar form, the backward map (output pixel → source pixel) is:
  r_source = r · exp(β·θ)
  θ_source = θ − β·ln(max(r, 1))
where β = ln(scale) / (2π).

With scale=256 (Escher's value) the picture repeats itself rotated by 157.6°
and shrunk by a factor of 22.58 — the structure of 'Print Gallery' (1956).

With --tile, the source image tiles as a repeating Cartesian grid and the
conformal warp bends that grid — adjacent tiles stay connected at their edges.

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

logger = logging.getLogger(__name__)


def _bilinear(arr: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Bilinear interpolation. Clamps out-of-bounds coords to array edges."""
    h, w = arr.shape[:2]
    x0 = np.floor(x_coords).astype(np.int32).clip(0, w - 1)
    x1 = (x0 + 1).clip(0, w - 1)
    y0 = np.floor(y_coords).astype(np.int32).clip(0, h - 1)
    y1 = (y0 + 1).clip(0, h - 1)
    wx = (x_coords - x0)[..., np.newaxis]
    wy = (y_coords - y0)[..., np.newaxis]
    result = ((1 - wy) * ((1 - wx) * arr[y0, x0] + wx * arr[y0, x1]) +
              wy       * ((1 - wx) * arr[y1, x0] + wx * arr[y1, x1]))
    return result.clip(0, 255).astype(np.uint8)


def _bilinear_wrap(arr: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Bilinear interpolation with toroidal (modular) wrapping."""
    h, w = arr.shape[:2]
    x_f = x_coords % w
    y_f = y_coords % h
    x0 = np.floor(x_f).astype(np.int32) % w
    x1 = (x0 + 1) % w
    y0 = np.floor(y_f).astype(np.int32) % h
    y1 = (y0 + 1) % h
    wx = (x_f - np.floor(x_f))[..., np.newaxis]
    wy = (y_f - np.floor(y_f))[..., np.newaxis]
    result = ((1 - wy) * ((1 - wx) * arr[y0, x0] + wx * arr[y0, x1]) +
              wy       * ((1 - wx) * arr[y1, x0] + wx * arr[y1, x1]))
    return result.clip(0, 255).astype(np.uint8)


def log_spiral_warp(img_arr: np.ndarray, scale: float = 256.0,
                    tile: bool = False,
                    cx: float = None, cy: float = None) -> np.ndarray:
    """Apply Escher/Lenstra conformal warp (complex power map w^α).

    α = (2πi + ln(scale)) / (2πi)

    Backward map in polar coords:
      r_in    = r · exp(β·θ)
      θ_in    = θ − β·ln(max(r, 1))
    where β = ln(scale) / (2π).

    With tile=True, source coordinates wrap modulo image size so the source
    tiles as a repeating Cartesian grid. The conformal warp bends that grid;
    adjacent tiles remain connected at their edges.

    Args:
        img_arr: uint8 (H, W, 3) source image
        scale:   self-similarity scale factor (Escher: 256)
        tile:    tile source as Cartesian grid before warping
        cx, cy:  centre of warp (defaults to image centre)

    Returns:
        uint8 (H, W, 3) warped image
    """
    beta = math.log(scale) / (2.0 * math.pi)

    h, w = img_arr.shape[:2]
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    r_in = r * np.exp(beta * theta)
    theta_in = theta - beta * np.log(np.maximum(r, 1.0))

    x_in = cx + r_in * np.cos(theta_in)
    y_in = cy + r_in * np.sin(theta_in)

    if tile:
        return _bilinear_wrap(img_arr, x_in, y_in)
    else:
        return _bilinear(img_arr, x_in, y_in)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Escher log-spiral warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./print-gallery-bot-output"))
    parser.add_argument("--scale", type=float, default=256.0,
                        help="Self-similarity scale factor (Escher: 256)")
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

    beta = math.log(args.scale) / (2.0 * math.pi)
    logger.info(f"Warp params: scale={args.scale}, β={beta:.4f}, tile={args.tile}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, warping...")
        arr = np.array(img)
        warped = log_spiral_warp(arr, scale=args.scale, tile=args.tile)
        dest = out_dir / f"print_gallery_result_{i}.png"
        Image.fromarray(warped).save(dest)
        logger.info(f"  Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="print-gallery-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
