"""Convert a raster image to a lossless, print-ready PDF.

The image is embedded at its original pixel dimensions.  DPI only controls
the physical page size; it does not resample or reduce the source image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def convert_image_to_pdf(source: Path, destination: Path, dpi: float = 300.0) -> None:
    """Embed *source* losslessly into a one-page PDF at the requested DPI."""
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")

    with Image.open(source) as image:
        image.load()
        width_px, height_px = image.size

        # PDF coordinates use points (72 points = 1 inch). No image resizing is
        # performed: the full source pixel grid is passed to ReportLab.
        width_pt = width_px * 72.0 / dpi
        height_pt = height_px * 72.0 / dpi

        # RGB/greyscale images avoid unnecessary alpha masks. Transparency is
        # retained for sources that genuinely use it.
        if image.mode == "RGBA" and image.getchannel("A").getextrema() == (255, 255):
            image = image.convert("RGB")
        elif image.mode not in {"1", "L", "RGB", "RGBA", "CMYK"}:
            image = image.convert("RGBA")

        destination.parent.mkdir(parents=True, exist_ok=True)
        pdf = canvas.Canvas(
            str(destination),
            pagesize=(width_pt, height_pt),
            pageCompression=1,
            invariant=1,
        )
        pdf.setTitle(source.stem)
        pdf.drawImage(
            ImageReader(image),
            0,
            0,
            width=width_pt,
            height=height_pt,
            preserveAspectRatio=True,
            mask="auto",
        )
        pdf.showPage()
        pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image to a lossless, one-page PDF."
    )
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to the output PDF")
    parser.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="Physical print resolution (default: 300); pixels are never resampled",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_image_to_pdf(args.input.resolve(), args.output.resolve(), args.dpi)
    print(f"Created: {args.output.resolve()}")
