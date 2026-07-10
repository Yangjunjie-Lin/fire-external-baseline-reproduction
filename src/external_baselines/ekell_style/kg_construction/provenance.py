from __future__ import annotations

from dataclasses import asdict, dataclass

from external_baselines.common.checksums import sha256_text


@dataclass(frozen=True)
class Provenance:
    source_id: str
    chunk_id: str
    source_text: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.chunk_id.strip():
            raise ValueError("source_id and chunk_id must be non-empty")
        if not self.source_text.strip():
            raise ValueError("source_text must be non-empty")

    @property
    def source_text_sha256(self) -> str:
        return sha256_text(self.source_text)

    def to_dict(self) -> dict[str, str]:
        return {**asdict(self), "source_text_sha256": self.source_text_sha256}


def provenance_from_segment(segment: object) -> Provenance:
    return Provenance(
        source_id=str(getattr(segment, "source_id")),
        chunk_id=str(getattr(segment, "chunk_id")),
        source_text=str(getattr(segment, "text")),
    )
