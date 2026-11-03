from specrag.chunkers import FixedWindowChunker, StructureAwareMarkdownChunker

SAMPLE_MARKDOWN = """# Hardware Specification

## Section 1: Overview
This is an introductory overview of the processor architecture.

## Section 2: Performance Matrix
Below is the official specification matrix:

| Metric | Mode A | Mode B |
| :--- | :--- | :--- |
| Clock Speed | 1.2 GHz | 2.5 GHz |
| Peak Power | 15 W | 45 W |
| Bandwidth | 25 GB/s | 100 GB/s |

## Section 3: Summary
End of the specification document.
"""


def test_fixed_window_chunker_basic():
    chunker = FixedWindowChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(SAMPLE_MARKDOWN, doc_id="test_doc.md")

    assert len(chunks) > 1
    assert all(c.doc_id == "test_doc.md" for c in chunks)
    assert chunks[0].section_breadcrumbs == []


def test_structure_aware_chunker_table_preservation():
    chunker = StructureAwareMarkdownChunker()
    chunks = chunker.chunk_document(SAMPLE_MARKDOWN, doc_id="test_doc.md")

    # Find the table chunk
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1

    table_chunk = table_chunks[0]
    # Verify the whole table is intact with all rows
    assert "| Clock Speed | 1.2 GHz | 2.5 GHz |" in table_chunk.text
    assert "| Peak Power | 15 W | 45 W |" in table_chunk.text
    assert "| Bandwidth | 25 GB/s | 100 GB/s |" in table_chunk.text

    # Verify hierarchical breadcrumbs are attached
    assert table_chunk.section_breadcrumbs == [
        "Hardware Specification",
        "Section 2: Performance Matrix",
    ]
    assert (
        "[Doc: test_doc.md > Hardware Specification > Section 2: Performance Matrix]"
        in table_chunk.contextualized_text
    )
