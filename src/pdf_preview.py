"""
PDF page preview: renders a cited page as a PNG image using PyMuPDF.
Used by the Streamlit UI to show the exact source page.
"""

import logging
from io import BytesIO
from pathlib import Path

import pymupdf as fitz

from src.config import PDF_PATH

logger = logging.getLogger(__name__)


def render_page_image(page_num: int, dpi: int = 150) -> bytes:
    """
    Render a specific page of the HR Manual as a PNG image.

    Args:
        page_num: 1-indexed page number.
        dpi: Resolution for rendering (default 150).

    Returns:
        PNG image as bytes.

    Raises:
        FileNotFoundError: If the PDF is not found.
        ValueError: If page_num is out of range.
    """
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if page_num < 1 or page_num > total_pages:
        doc.close()
        raise ValueError(
            f"Page {page_num} out of range. PDF has {total_pages} pages."
        )

    page = doc[page_num - 1]  # 0-indexed internally
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()

    logger.debug("Rendered page %d as PNG (%d bytes)", page_num, len(img_bytes))
    return img_bytes


# Alias for compatibility
render_page_as_image = render_page_image
