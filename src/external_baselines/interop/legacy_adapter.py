"""Deprecated pre-canonical helpers retained for diagnostics only.

Do not use in the formal firebench-interop-v1 adapter path.
Formal conversion uses interop.normalizer + taxonomy IDs.
"""

from __future__ import annotations

from typing import Any

from external_baselines.common.text_utils import as_list


def action_objects_legacy(actions: Any) -> list[dict[str, Any]]:
    """Deprecated: invents action_N IDs. Prefer interop.normalizer."""
    out: list[dict[str, Any]] = []
    for i, item in enumerate(as_list(actions)):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("action") or item.get("description") or "")
            action_id = str(item.get("action_id") or item.get("id") or f"action_{i+1}")
            priority = item.get("priority")
            refs = as_list(item.get("evidence_refs") or [])
            out.append({
                "action_id": action_id,
                "text": text,
                "priority": str(priority) if priority is not None else None,
                "evidence_refs": [str(r) for r in refs],
            })
        else:
            out.append({
                "action_id": f"action_{i+1}",
                "text": str(item),
                "priority": None,
                "evidence_refs": [],
            })
    return out


def blocked_objects_legacy(items: Any) -> list[dict[str, Any]]:
    """Deprecated: object-shaped blocked actions. Formal schema uses string IDs."""
    out: list[dict[str, Any]] = []
    for i, item in enumerate(as_list(items)):
        if isinstance(item, dict):
            out.append({
                "action_id": str(item.get("action_id") or item.get("id") or f"blocked_{i+1}"),
                "text": str(item.get("text") or item.get("action") or item),
            })
        else:
            out.append({"action_id": f"blocked_{i+1}", "text": str(item)})
    return out
