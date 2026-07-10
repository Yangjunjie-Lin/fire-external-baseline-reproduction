from __future__ import annotations

import re
from dataclasses import dataclass

from external_baselines.common.checksums import sha256_text


@dataclass(frozen=True)
class DocumentSegment:
    source_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    text_sha256: str


def parse_document(
    text: str,
    *,
    source_id: str,
    max_chars: int = 1200,
    overlap_chars: int = 0,
) -> list[DocumentSegment]:
    """Split document text on paragraphs/sentences while preserving source offsets."""
    if not source_id.strip():
        raise ValueError("source_id must be non-empty")
    if max_chars < 1 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("require max_chars > overlap_chars >= 0")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    spans = [
        (match.start(), match.end(), match.group().strip())
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", normalized, re.DOTALL)
        if match.group().strip()
    ]
    if not spans and normalized.strip():
        start = len(normalized) - len(normalized.lstrip())
        spans = [(start, len(normalized.rstrip()), normalized.strip())]

    chunks: list[tuple[int, int, str]] = []
    for start, end, paragraph in spans:
        cursor = 0
        while cursor < len(paragraph):
            stop = min(len(paragraph), cursor + max_chars)
            if stop < len(paragraph):
                boundary = max(
                    paragraph.rfind(". ", cursor, stop),
                    paragraph.rfind("。", cursor, stop),
                    paragraph.rfind(" ", cursor, stop),
                )
                if boundary > cursor:
                    stop = boundary + 1
            piece = paragraph[cursor:stop].strip()
            if piece:
                local = paragraph.find(piece, cursor)
                chunks.append((start + local, start + local + len(piece), piece))
            if stop >= len(paragraph):
                break
            cursor = max(cursor + 1, stop - overlap_chars)

    return [
        DocumentSegment(
            source_id=source_id,
            chunk_id=f"{source_id}:chunk:{index:04d}",
            text=piece,
            start_char=start,
            end_char=end,
            text_sha256=sha256_text(piece),
        )
        for index, (start, end, piece) in enumerate(chunks)
    ]


parse_text = parse_document
