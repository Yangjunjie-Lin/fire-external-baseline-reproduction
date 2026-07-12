from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .schema import KGTriple, ReviewStatus


def partition_review_queue(triples: Iterable[KGTriple]) -> dict[ReviewStatus, list[KGTriple]]:
    queues: dict[ReviewStatus, list[KGTriple]] = {
        "candidate": [], "approved": [], "rejected": [],
    }
    for triple in triples:
        queues[triple.review_status].append(triple)
    return queues


def set_review_status(
    triple: KGTriple,
    status: ReviewStatus,
    *,
    reviewer: str | None = None,
    review_note: str | None = None,
    human_reviewed: bool = False,
) -> KGTriple:
    """Apply an explicit review decision without pretending automation is human review."""
    if status in {"approved", "rejected"} and not human_reviewed:
        raise ValueError("approved/rejected requires human_reviewed=True")
    attributes = dict(triple.attributes)
    if reviewer:
        attributes["reviewer"] = reviewer
    if review_note:
        attributes["review_note"] = review_note
    attributes["human_reviewed"] = bool(human_reviewed)
    return replace(triple, review_status=status, attributes=attributes)


def pending_candidates(triples: Iterable[KGTriple]) -> list[KGTriple]:
    return [triple for triple in triples if triple.review_status == "candidate"]
