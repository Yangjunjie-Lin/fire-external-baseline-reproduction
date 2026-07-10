from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Iterable

from external_baselines.common.checksums import sha256_text
from external_baselines.common.llm_client import LLMClient

from .document_parser import DocumentSegment
from .schema import KGTriple


def _triple(
    segment: DocumentSegment,
    subject: str,
    predicate: str,
    object_: str,
    *,
    method: str,
    confidence: float,
) -> KGTriple:
    identity = "\x1f".join((segment.source_id, segment.chunk_id, subject, predicate, object_))
    return KGTriple(
        triple_id=f"candidate:{sha256_text(identity)[:20]}",
        subject=subject.strip(),
        predicate=predicate.strip(),
        object=object_.strip(),
        source_id=segment.source_id,
        chunk_id=segment.chunk_id,
        source_text=segment.text,
        extraction_method=method,
        confidence=max(0.0, min(1.0, float(confidence))),
        review_status="candidate",
    )


def heuristic_extract(segment: DocumentSegment) -> list[KGTriple]:
    """Extract conservative candidates from explicit subject-relation-object clauses."""
    patterns = (
        (r"(?P<s>[^.;。\n]{1,100}?)\s+(?:requires|needs)\s+(?P<o>[^.;。\n]{1,120})", "requires"),
        (r"(?P<s>[^.;。\n]{1,100}?)\s+(?:affects|threatens)\s+(?P<o>[^.;。\n]{1,120})", "affects"),
        (r"(?P<s>[^.;。\n]{1,100}?)\s+(?:occurs at|is located at)\s+(?P<o>[^.;。\n]{1,120})", "occurs_at"),
        (r"(?P<s>[^.;。\n]{1,100}?)\s+(?:uses)\s+(?P<o>[^.;。\n]{1,120})", "uses"),
    )
    output: list[KGTriple] = []
    for pattern, predicate in patterns:
        for match in re.finditer(pattern, segment.text, flags=re.IGNORECASE):
            output.append(
                _triple(
                    segment, match.group("s"), predicate, match.group("o"),
                    method="heuristic_pattern", confidence=0.55,
                )
            )
    return list({item.triple_id: item for item in output}.values())


def _json_array(raw: str) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def extract_candidate_triples(
    segments: Iterable[DocumentSegment],
    *,
    llm: LLMClient | None = None,
) -> list[KGTriple]:
    """Return review candidates only; outputs are never official E-KELL triples."""
    candidates: list[KGTriple] = []
    for segment in segments:
        if llm is None:
            candidates.extend(heuristic_extract(segment))
            continue
        prompt = (
            "Extract only explicit candidate KG statements from SOURCE_TEXT. Return a JSON "
            "array of objects with subject, predicate, object, confidence. Do not infer missing "
            "facts and do not call these official E-KELL triples.\n"
            f"SOURCE_ID: {segment.source_id}\nCHUNK_ID: {segment.chunk_id}\n"
            f"SOURCE_TEXT:\n{segment.text}"
        )
        raw = llm.complete(
            system="You produce conservative, provenance-preserving candidate triples as JSON.",
            user=prompt,
            temperature=0.0,
            max_tokens=1200,
        )
        rows = _json_array(raw)
        if not rows:
            candidates.extend(heuristic_extract(segment))
            continue
        for row in rows:
            subject, predicate, object_ = (
                str(row.get("subject", "")).strip(),
                str(row.get("predicate", "")).strip(),
                str(row.get("object", "")).strip(),
            )
            if subject and predicate and object_:
                candidates.append(
                    _triple(
                        segment, subject, predicate, object_,
                        method="llm_assisted_candidate",
                        confidence=float(row.get("confidence", 0.5)),
                    )
                )
    return [replace(item, review_status="candidate") for item in candidates]


extract_triples = extract_candidate_triples
