"""Chunking strategies for SpecRAG.

Includes:
- FixedWindowChunker (Phase A: Baseline naive character chunking that splits tables)
- StructureAwareMarkdownChunker (Phase B: Hierarchical header-aware + atomic table preservation)
"""

import re

from specrag.schemas import DocumentChunk


class FixedWindowChunker:
    """Naive fixed-window chunker that splits text blindly without table or header awareness."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Split text blindly into fixed-size windows."""
        chunks: list[DocumentChunk] = []
        start = 0
        text_len = len(text)
        index = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]

            # Detect if this slice arbitrarily sliced into a markdown table
            is_table_slice = "|" in chunk_text and "\n|---" in chunk_text

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_naive_{index:04d}",
                    doc_id=doc_id,
                    text=chunk_text,
                    is_table=is_table_slice,
                    section_breadcrumbs=[],
                    metadata={"chunk_index": index, "chunk_strategy": "fixed_window"},
                )
            )

            index += 1
            start += self.chunk_size - self.chunk_overlap
            if start >= text_len:
                break

        return chunks


class StructureAwareMarkdownChunker:
    """Structure-aware chunker that tracks header hierarchies and keeps tables intact."""

    def __init__(self, max_prose_chunk_size: int = 1200):
        self.max_prose_chunk_size = max_prose_chunk_size

    def _is_table_line(self, line: str) -> bool:
        s = line.strip()
        return s.startswith("|") and s.endswith("|")

    def chunk_document(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Parse Markdown sections, keeping tables intact and attaching hierarchical breadcrumbs."""
        lines = text.splitlines()
        chunks: list[DocumentChunk] = []

        current_headers: dict[int, str] = {}  # level -> title
        current_prose: list[str] = []
        in_table = False
        current_table: list[str] = []
        chunk_idx = 0

        def get_breadcrumbs() -> list[str]:
            return [current_headers[k] for k in sorted(current_headers.keys())]

        def flush_prose():
            nonlocal chunk_idx, current_prose
            if not current_prose:
                return
            prose_text = "\n".join(current_prose).strip()
            if prose_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_struct_{chunk_idx:04d}",
                        doc_id=doc_id,
                        text=prose_text,
                        is_table=False,
                        section_breadcrumbs=get_breadcrumbs(),
                        metadata={"chunk_index": chunk_idx, "chunk_strategy": "structure_aware"},
                    )
                )
                chunk_idx += 1
            current_prose = []

        def flush_table():
            nonlocal chunk_idx, current_table
            if not current_table:
                return
            table_text = "\n".join(current_table).strip()
            if table_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_table_{chunk_idx:04d}",
                        doc_id=doc_id,
                        text=table_text,
                        is_table=True,
                        section_breadcrumbs=get_breadcrumbs(),
                        metadata={
                            "chunk_index": chunk_idx,
                            "chunk_strategy": "atomic_table",
                            "row_count": len(current_table),
                        },
                    )
                )
                chunk_idx += 1
            current_table = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())

            if header_match:
                if in_table:
                    flush_table()
                    in_table = False
                flush_prose()

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Pop any deeper header levels
                current_headers = {k: v for k, v in current_headers.items() if k < level}
                current_headers[level] = title
                continue

            if self._is_table_line(line):
                if not in_table:
                    # Strip trailing whitespace/blank lines from current_prose first
                    while current_prose and not current_prose[-1].strip():
                        current_prose.pop()

                    # Check if preceding line is a table caption/title (e.g. 'Table 1. ...' or '##### Table 2. ...')
                    table_caption = None
                    if current_prose and re.match(
                        r"^(#{1,6}\s+)?Table\s+\d+", current_prose[-1].strip(), re.I
                    ):
                        table_caption = current_prose.pop().strip()
                    flush_prose()
                    in_table = True
                    if table_caption:
                        current_table.append(table_caption)
                current_table.append(line)
            else:
                if in_table:
                    flush_table()
                    in_table = False
                current_prose.append(line)
                if sum(len(line_item) for line_item in current_prose) > self.max_prose_chunk_size:
                    flush_prose()

        # Flush any trailing blocks
        if in_table:
            flush_table()
        flush_prose()

        return chunks
