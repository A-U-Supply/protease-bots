"""Curl noise warp bot.

Fetches 3 images and applies a divergence-free (curl noise) displacement
field as a backward warp.

The displacement field is the curl of a smooth random scalar potential φ:
  u(x,y) = ( ∂φ/∂y , −∂φ/∂x )

Because it is the curl of a scalar, u is exactly divergence-free — swirling
and incompressible, like an idealised 2D fluid.  Pixels appear to have been
pushed through invisible eddies and vortices.

φ is built as fractal Brownian motion: `octaves` layers of Gaussian-smoothed
white noise, each at half the spatial scale of the previous, summed with
halving amplitude.  Smoothing is done via FFT so no extra dependencies are
needed beyond numpy.

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


def _gaussian_blur_fft(field: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 2-D field with a Gaussian kernel via FFT (no scipy needed).

    Args:
        field: (H, W) float64 array
        sigma: Gaussian standard deviation in pixels

    Returns:
        (H, W) float64 smoothed array
    """
    h, w = field.shape
    ky = np.fft.fftfreq(h)
    kx = np.fft.fftfreq(w)
    kx2d, ky2d = np.meshgrid(kx, ky)
    kernel = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (kx2d ** 2 + ky2d ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * kernel))


def curl_noise_warp(img_arr: np.ndarray,
                    strength: float = 120.0,
                    scale: float = 0.20,
                    octaves: int = 4,
                    seed=None,
                    tile: bool = False) -> np.ndarray:
    """Apply a curl-noise backward warp to an image.

    Builds a fractal Brownian motion scalar potential then takes its 2-D curl
    to get a divergence-free displacement field.  Each output pixel is mapped
    back to a source pixel displaced by that field, producing swirling,
    fluid-like distortion.

    Args:
        img_arr:  uint8 (H, W, 3) source image
        strength: maximum displacement in pixels (default 120)
        scale:    spatial scale of the broadest noise octave as a fraction of
                  the shorter image dimension (default 0.20 → 20 %)
        octaves:  number of fBm octaves; more = finer detail (default 4)
        seed:     random seed; None = different result each run
        tile:     wrap source image at boundaries instead of clamping

    Returns:
        uint8 (H, W, 3) warped image
    """
    h, w = img_arr.shape[:2]
    rng = np.random.default_rng(seed)

    sigma_base = scale * min(w, h)

    # Build fractal Brownian motion potential φ
    potential = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    total_amp = 0.0
    sig = sigma_base
    for _ in range(octaves):
        if sig < 1.0:
            break
        noise = rng.standard_normal((h, w))
        potential += amplitude * _gaussian_blur_fft(noise, sig)
        total_amp += amplitude
        sig *= 0.5
        amplitude *= 0.5

    if total_amp > 0:
        potential /= total_amp

    # Curl in 2-D: vx = ∂φ/∂y, vy = −∂φ/∂x
    grad_y, grad_x = np.gradient(potential)
    disp_x = grad_y
    disp_y = -grad_x

    # Normalise so the 99th-percentile magnitude equals `strength`
    mag = np.sqrt(disp_x ** 2 + disp_y ** 2)
    p99 = np.percentile(mag, 99)
    if p99 > 0:
        disp_x = disp_x / p99 * strength
        disp_y = disp_y / p99 * strength

    # Backward map: for each output pixel, sample source at displaced position
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    x_in = xx - disp_x
    y_in = yy - disp_y

    if tile:
        return _bilinear_wrap(img_arr, x_in, y_in)
    else:
        return _bilinear(img_arr, x_in, y_in)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Curl noise (divergence-free) warp bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./curl-noise-warp-bot-output"))
    parser.add_argument("--strength", type=float, default=120.0,
                        help="Maximum displacement in pixels (default 120)")
    parser.add_argument("--scale", type=float, default=0.20,
                        help="Broadest noise octave as fraction of short image dimension "
                             "(default 0.20)")
    parser.add_argument("--octaves", type=int, default=4,
                        help="fBm octaves; more = finer swirl detail (default 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; omit for a different result each run")
    parser.add_argument("--tile", action="store_true",
                        help="Wrap source at boundaries instead of clamping to edge colour")
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

    logger.info(f"Warp params: strength={args.strength}, scale={args.scale}, "
                f"octaves={args.octaves}, seed={args.seed}, tile={args.tile}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        logger.info(f"Image {i}: {img.size[0]}×{img.size[1]}, warping...")
        arr = np.array(img)
        warped = curl_noise_warp(
            arr,
            strength=args.strength,
            scale=args.scale,
            octaves=args.octaves,
            seed=args.seed,
            tile=args.tile,
        )
        dest = out_dir / f"curl_noise_result_{i}.png"
        Image.fromarray(warped).save(dest)
        logger.info(f"  Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="curl-noise-warp-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
