from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .text_utils import tokenize


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        ...


@dataclass
class HeuristicLLMClient:
    """Deterministic fallback client for reproducible smoke tests.

    It deliberately does not implement SAFE-like gating. It only emits a normalized
    emergency-response-shaped JSON object based on keywords present in the prompt.
    """

    model: str = "local-deterministic-baseline"

    def complete(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        text = f"{system}\n{user}".lower()
        risks: list[str] = []
        actions: list[str] = []
        missing: list[str] = []
        blocked: list[str] = []
        evidence: list[str] = []
        citations: list[str] = []

        if "electrical" in text or "power" in text:
            risks.append("Possible electrical fire or energized equipment hazard.")
            actions.append("Confirm power isolation status before selecting suppression actions.")
            missing.append("Power cutoff / electrical isolation status.")
            blocked.append("Avoid unsupported water-based suppression until electrical isolation is confirmed.")
        if "smoke" in text:
            risks.append("Smoke exposure may reduce visibility and create respiratory risk.")
            actions.append("Use respiratory protection and controlled entry procedures if entry is required.")
            missing.append("Respiratory protection readiness and entry-team status.")
            blocked.append("Avoid entry without confirmed respiratory protection in high-smoke conditions.")
        if "crowd" in text or "evac" in text or "shopping mall" in text or "mall" in text:
            risks.append("Public occupancy may require evacuation coordination and crowd-control support.")
            actions.append("Coordinate evacuation route confirmation and public-area isolation.")
        if "chemical" in text or "hazmat" in text or "gas" in text:
            risks.append("Potential hazardous-material or gas-related escalation risk.")
            actions.append("Identify hazardous materials and isolate the affected area.")
            missing.append("Hazardous-material identity and concentration status.")
        if not risks:
            risks.append("Insufficient scenario detail for reliable emergency risk characterization.")
            missing.append("Incident type, location, hazards, affected people, and resource status.")
            actions.append("Collect missing incident details and consult qualified incident command.")

        has_supplied_context = (
            "retrieved contexts:" in user.lower()
            or "retrieved kg facts" in user.lower()
            or "graph/text contexts:" in user.lower()
            or "kg facts and evidence:" in user.lower()
        ) and "(none)" not in user.lower()
        if has_supplied_context:
            evidence.append("Retrieved contexts or KG facts were considered where provided by the baseline pipeline.")
            found = sorted(set(re.findall(r"(?:citation|source_id|chunk_id)[:=]\s*([A-Za-z0-9_:\-./]+)", user)))
            citations.extend(found[:5])

        if "prompt 1: situation understanding" in user.lower():
            return json.dumps({
                "incident_type": "electrical_fire" if "electrical" in text else "unspecified_fire_emergency",
                "involved_entities": [x for x in ["electrical_fire" if "electrical" in text else "", "high_smoke" if "smoke" in text else "", "shopping_mall" if "mall" in text else ""] if x],
                "hazards": risks,
                "emergency_stage": "initial_response",
                "information_gaps": list(dict.fromkeys(missing)),
                "citations": citations,
            }, ensure_ascii=False)

        if "prompt 2: kg-grounded decision reasoning" in user.lower():
            return json.dumps({
                "reasoning_summary": "Actions are proposed only from the scenario and retrieved KG/evidence context supplied to this baseline.",
                "candidate_actions": actions,
                "unsupported_or_deferred_actions": blocked,
                "missing_information": list(dict.fromkeys(missing)),
                "citations": citations,
            }, ensure_ascii=False)

        if missing or blocked:
            gate = "critical_risk_requires_human_confirmation"
        else:
            gate = "baseline_response_without_explicit_gate"

        payload = {
            "situation_summary": "Baseline-generated emergency decision-support summary based on the provided scenario text and any supplied context.",
            "key_risks": risks,
            "recommended_actions": actions,
            "blocked_or_unsafe_actions": blocked,
            "missing_confirmations": list(dict.fromkeys(missing)),
            "supporting_evidence": evidence,
            "citations": citations,
            "final_decision_gate": gate,
        }

        if "incident_type" in user and "information_gaps" in user and "scenario parsing" in user.lower():
            payload = {
                "incident_type": "electrical_fire" if "electrical" in text else "unspecified_fire_emergency",
                "location": "shopping_mall" if "mall" in text else "unspecified_location",
                "hazards": [r for r in ["high_smoke" if "smoke" in text else "", "energized_equipment" if "electrical" in text else ""] if r],
                "affected_people": ["public_occupants"] if "mall" in text or "crowd" in text else [],
                "emergency_stage": "initial_response",
                "information_gaps": list(dict.fromkeys(missing)),
            }
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class OpenAIChatClient:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"

    def complete(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed. Use heuristic provider or install openai.") from exc
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set.")
        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv(self.base_url_env)
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def build_llm_client(config: dict[str, Any] | None = None) -> LLMClient:
    config = config or {}
    llm_cfg = config.get("llm", config) if isinstance(config, dict) else {}
    provider = str(llm_cfg.get("provider", "heuristic")).lower()
    model = str(llm_cfg.get("model", "local-deterministic-baseline"))
    if provider in {"openai", "openai_chat"}:
        return OpenAIChatClient(
            model=model,
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url_env=str(llm_cfg.get("base_url_env", "OPENAI_BASE_URL")),
        )
    return HeuristicLLMClient(model=model)
