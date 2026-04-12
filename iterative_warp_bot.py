"""Iterative conformal warp bot.

Fetches 3 images. Applies the Escher/Lenstra log-spiral conformal warp
repeatedly, feeding each output as the input to the next pass. Posts all
N iterations for each image — the full progression from pass 1 to pass N.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from print_gallery_bot import log_spiral_warp

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Iterative conformal warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./iterative-warp-bot-output"))
    parser.add_argument("--scale", type=float, default=256.0,
                        help="Self-similarity scale factor (Escher: 256)")
    parser.add_argument("--iterations", type=int, default=4,
                        help="Number of warp passes (default 4)")
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

    logger.info(f"Warp params: scale={args.scale}, iterations={args.iterations}, tile={args.tile}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}")
        arr = np.array(img)
        for n in range(1, args.iterations + 1):
            arr = log_spiral_warp(arr, scale=args.scale, tile=args.tile)
            dest = out_dir / f"iterative_result_{i}_iter_{n}.png"
            Image.fromarray(arr).save(dest)
            logger.info(f"  Pass {n} → {dest.name}")
            output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="iterative-warp-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
