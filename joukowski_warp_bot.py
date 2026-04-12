"""Joukowski conformal warp bot.

Fetches 3 images and applies the Joukowski map as a backward warp:
  source = z + 1/z

where z = (pixel - centre) / norm + shift, and norm = (min_dim/2) / scale.

The Joukowski transform (classical aerodynamics map) creates:
  - A singularity vortex at z = 0 (epsilon-protected)
  - Compression inside the unit circle, expansion outside
  - Fold lines at the critical points z = ±1 (where dw/dz = 0)
  - With --tile: interlocking lens/crescent shapes from the folding

The --shift-r parameter moves the singularity off-centre horizontally,
producing an asymmetric off-axis vortex.

Posts 3 warped images.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from print_gallery_bot import _bilinear, _bilinear_wrap

logger = logging.getLogger(__name__)


def joukowski_warp(img_arr: np.ndarray, scale: float = 1.5,
                   shift_r: float = 0.3, shift_i: float = 0.0,
                   tile: bool = False,
                   cx: float = None, cy: float = None) -> np.ndarray:
    """Apply Joukowski conformal warp (backward map: source = z + 1/z).

    Pixel coordinates are normalised so the short half-side of the image
    spans `scale` units. The backward map samples the source at w = z + 1/z
    for each output pixel z (after applying the complex shift).

    In Cartesian form:
      w_real = x(1 + 1/r²)
      w_imag = y(1 - 1/r²)
    where r² = x² + y² and the singularity at r=0 is epsilon-protected.

    Args:
        img_arr:  uint8 (H, W, 3) source image
        scale:    units spanned by the short half-side (default 1.5).
                  Controls where the unit circle falls — at 1.5 the unit
                  circle is at 2/3 of the image half-side.
        shift_r:  real part of the singularity offset (default 0.3).
                  Moves the vortex left of centre.
        shift_i:  imaginary part of the singularity offset (default 0.0).
        tile:     tile source as Cartesian grid before warping
        cx, cy:   centre of warp (defaults to image centre)

    Returns:
        uint8 (H, W, 3) warped image
    """
    h, w = img_arr.shape[:2]
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    norm = min(w, h) / 2.0 / scale  # pixels per unit

    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    zr = (xx - cx) / norm + shift_r
    zi = (yy - cy) / norm + shift_i

    r2 = np.maximum(zr ** 2 + zi ** 2, 1e-6)

    # w = z + 1/z  →  1/z = conj(z)/|z|²
    src_r = zr + zr / r2   # w_real = x(1 + 1/r²)
    src_i = zi - zi / r2   # w_imag = y(1 - 1/r²)

    # Undo the shift before converting back to pixel coords
    x_in = cx + (src_r - shift_r) * norm
    y_in = cy + (src_i - shift_i) * norm

    if tile:
        return _bilinear_wrap(img_arr, x_in, y_in)
    else:
        return _bilinear(img_arr, x_in, y_in)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Joukowski conformal warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./joukowski-warp-bot-output"))
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Units spanned by short half-side (default 1.5)")
    parser.add_argument("--shift-r", type=float, default=0.3,
                        help="Real part of singularity offset (default 0.3)")
    parser.add_argument("--shift-i", type=float, default=0.0,
                        help="Imaginary part of singularity offset (default 0.0)")
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

    logger.info(f"Warp params: scale={args.scale}, "
                f"shift=({args.shift_r:.3f}+{args.shift_i:.3f}j), tile={args.tile}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, warping...")
        arr = np.array(img)
        warped = joukowski_warp(arr, scale=args.scale,
                                shift_r=args.shift_r, shift_i=args.shift_i,
                                tile=args.tile)
        dest = out_dir / f"joukowski_result_{i}.png"
        Image.fromarray(warped).save(dest)
        logger.info(f"  Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="joukowski-warp-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
