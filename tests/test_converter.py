from pathlib import Path

import pytest

from specrag.ingestion.converter import DocumentConverter


def test_converter_initialization():
    converter = DocumentConverter()
    assert converter.md is not None
    assert converter.use_pymupdf_for_pdf is True


def test_converter_file_not_found():
    converter = DocumentConverter()
    with pytest.raises(FileNotFoundError):
        converter.convert_file("non_existent_file.pdf")


def test_converter_existing_markdown_file(tmp_path: Path):
    test_file = tmp_path / "sample.md"
    test_file.write_text("# Test Header\n\nSome text content.", encoding="utf-8")

    converter = DocumentConverter()
    result = converter.convert_file(test_file)
    assert "Test Header" in result
    assert "Some text content" in result
