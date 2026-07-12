"""Paper-informed E-KELL-style KG construction scaffolding.

This package creates independent review candidates, not official E-KELL triples.
"""

from .document_parser import DocumentSegment, parse_document, parse_text
from .kg_builder import build_kg, build_knowledge_graph
from .provenance import Provenance
from .review_queue import partition_review_queue, pending_candidates, set_review_status
from .schema import DECISION_DEMANDS, SCHEMA_VERSION, KGTriple, schema_document
from .triple_extractor import extract_candidate_triples, extract_triples
from .triple_validator import TripleValidation, validate_triple, validate_triples

__all__ = [
    "DECISION_DEMANDS", "DocumentSegment", "KGTriple", "Provenance", "SCHEMA_VERSION",
    "TripleValidation", "build_kg", "build_knowledge_graph", "extract_candidate_triples",
    "extract_triples", "parse_document", "parse_text", "partition_review_queue",
    "pending_candidates", "schema_document", "set_review_status", "validate_triple",
    "validate_triples",
]
