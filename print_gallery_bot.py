"""Escher/Lenstra log-spiral conformal warp bot.

Fetches 3 images and applies the Escher/Lenstra log-spiral conformal map to each.
The warp is a complex power map f(z) = z^a where a = α + iβ. In Escher's case
(α=1), pixels at larger radii rotate more than pixels near the centre, producing
the characteristic inward spiral seen in 'Print Gallery' (1956).

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


def log_spiral_warp(img_arr: np.ndarray, alpha: float = 1.0, beta: float = None,
                    cx: float = None, cy: float = None) -> np.ndarray:
    """Apply Escher/Lenstra log-spiral conformal warp.

    Backward map: for each output pixel, compute source coordinates.
      r_in    = r_out^(1/alpha)
      theta_in = theta_out/alpha - (beta/alpha^2) * ln(max(r_out, 1))
      x_in    = cx + r_in * cos(theta_in)
      y_in    = cy + r_in * sin(theta_in)

    For alpha=1 (Escher): r_in = r_out, rotation grows logarithmically with radius.

    Args:
        img_arr: uint8 (H, W, 3) source image
        alpha:   radial scaling exponent (1.0 = no scaling)
        beta:    spiral rate; default 2π/ln(256) ≈ 1.133 (Escher's value)
        cx, cy:  centre of warp (defaults to image centre)

    Returns:
        uint8 (H, W, 3) warped image
    """
    if beta is None:
        beta = 2.0 * math.pi / math.log(256.0)

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
    r_out = np.sqrt(dx ** 2 + dy ** 2)
    theta_out = np.arctan2(dy, dx)

    r_in = r_out ** (1.0 / alpha)
    theta_in = theta_out / alpha - (beta / alpha ** 2) * np.log(np.maximum(r_out, 1.0))

    x_in = cx + r_in * np.cos(theta_in)
    y_in = cy + r_in * np.sin(theta_in)

    # Bilinear interpolation
    x0 = np.floor(x_in).astype(np.int32).clip(0, w - 1)
    x1 = (x0 + 1).clip(0, w - 1)
    y0 = np.floor(y_in).astype(np.int32).clip(0, h - 1)
    y1 = (y0 + 1).clip(0, h - 1)
    wx = (x_in - x0)[..., np.newaxis]
    wy = (y_in - y0)[..., np.newaxis]

    result = ((1 - wy) * ((1 - wx) * img_arr[y0, x0] + wx * img_arr[y0, x1]) +
              wy       * ((1 - wx) * img_arr[y1, x0] + wx * img_arr[y1, x1]))
    return result.clip(0, 255).astype(np.uint8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Escher log-spiral warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./print-gallery-bot-output"))
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Radial scaling exponent (1.0 = Escher case, no scaling)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Spiral rate (default: 2π/ln(256) ≈ 1.133, Escher's value)")
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

    beta = args.beta if args.beta is not None else 2.0 * math.pi / math.log(256.0)
    logger.info(f"Warp params: alpha={args.alpha}, beta={beta:.4f}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, warping...")
        arr = np.array(img)
        warped = log_spiral_warp(arr, alpha=args.alpha, beta=beta)
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
