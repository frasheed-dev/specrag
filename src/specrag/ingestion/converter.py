"""Document conversion pipeline using Microsoft MarkItDown and PyMuPDF4LLM."""

import logging
from pathlib import Path

import pymupdf4llm
from markitdown import MarkItDown

logger = logging.getLogger("specrag.ingestion")


class DocumentConverter:
    """Converts PDF and enterprise documents into structure-preserving Markdown."""

    def __init__(self, use_pymupdf_for_pdf: bool = True):
        self.md = MarkItDown()
        self.use_pymupdf_for_pdf = use_pymupdf_for_pdf

    def convert_file(self, file_path: str | Path) -> str:
        """Convert a document file to clean Markdown text.

        Preserves multi-column tables, heading levels, and formatting.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        # For PDFs, pymupdf4llm provides exceptional table and layout preservation for technical datasheets
        if suffix == ".pdf" and self.use_pymupdf_for_pdf:
            try:
                logger.info(f"Converting PDF with PyMuPDF4LLM: {path.name}")
                md_text = pymupdf4llm.to_markdown(str(path))
                if md_text and len(md_text.strip()) > 0:
                    return md_text
            except Exception as e:
                logger.warning(
                    f"PyMuPDF4LLM failed for {path.name}: {e}. Falling back to MarkItDown."
                )

        # MarkItDown conversion for all formats (docx, pptx, xlsx, html, pdf fallback)
        logger.info(f"Converting document with Microsoft MarkItDown: {path.name}")
        result = self.md.convert(str(path))
        return result.text_content

    def convert_and_save(self, file_path: str | Path, output_md_path: str | Path) -> Path:
        """Convert a document and save the Markdown to disk."""
        out = Path(output_md_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        md_text = self.convert_file(file_path)
        out.write_text(md_text, encoding="utf-8")
        logger.info(f"Saved converted Markdown to: {out}")
        return out
