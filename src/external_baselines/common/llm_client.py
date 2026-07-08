from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        ...


@dataclass
class HeuristicLLMClient:
    """Deterministic fallback client for reproducible smoke tests only.

    This client deliberately does not implement SAFE-like routing, safety checking,
    dynamic-state logic, HITL gates, or target-project risk scoring. It only emits
    stable JSON-shaped responses so the external baseline pipelines can be tested
    without network/model credentials.
    """

    model: str = "local-deterministic-heuristic-smoke-test"
    provider: str = "heuristic"
    heuristic_fallback: bool = True

    def complete(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        text = f"{system}\n{user}".lower()
        risks: list[str] = []
        actions: list[str] = []
        missing: list[str] = []
        blocked: list[str] = []
        evidence: list[str] = []
        citations: list[str] = []

        if "electrical" in text or "power" in text or "电" in text:
            risks.append("Possible electrical fire or energized equipment hazard.")
            actions.append("Confirm power isolation status before selecting suppression actions.")
            missing.append("Power cutoff / electrical isolation status.")
            blocked.append("Defer water-based suppression until electrical isolation is confirmed by qualified personnel.")
        if "smoke" in text or "烟" in text:
            risks.append("Smoke exposure may reduce visibility and create respiratory risk.")
            actions.append("Use respiratory protection and controlled entry procedures if entry is required.")
            missing.append("Respiratory protection readiness and entry-team status.")
            blocked.append("Defer interior entry unless respiratory protection and entry-team readiness are confirmed.")
        if "crowd" in text or "evac" in text or "shopping mall" in text or "mall" in text or "商场" in text:
            risks.append("Public occupancy may require evacuation coordination and crowd-control support.")
            actions.append("Coordinate evacuation route confirmation and public-area isolation.")
        if "chemical" in text or "hazmat" in text or "gas" in text or "化学" in text or "燃气" in text:
            risks.append("Potential hazardous-material or gas-related escalation risk.")
            actions.append("Identify hazardous materials and isolate the affected area.")
            missing.append("Hazardous-material identity and concentration status.")
        if not risks:
            risks.append("Insufficient scenario detail for reliable emergency risk characterization.")
            missing.append("Incident type, location, hazards, affected people, and resource status.")
            actions.append("Collect missing incident details and consult qualified incident command.")

        has_context = any(marker in user.lower() for marker in [
            "retrieved contexts:", "retrieved kg facts", "kg facts and evidence:", "graph/text contexts:"
        ]) and "(none)" not in user.lower()
        if has_context:
            evidence.append("Retrieved KG facts or evidence contexts were considered by this external baseline.")
            found = sorted(set(re.findall(r"(?:citation|source_id|chunk_id|context_id)[:=]\s*([A-Za-z0-9_:\-./]+)", user)))
            citations.extend(found[:8])

        # Scenario parser schema.
        if "scenario parsing task" in user.lower() or "parser output schema" in user.lower():
            payload = {
                "incident_type": "electrical_fire" if ("electrical" in text or "power" in text or "电" in text) else ("hazmat_fire" if "chemical" in text or "hazmat" in text or "化学" in text else "fire_emergency"),
                "location": "shopping_mall" if ("shopping mall" in text or "mall" in text or "商场" in text) else ("electrical_room" if "electrical room" in text else "unspecified_location"),
                "building_type": "shopping_mall" if ("shopping mall" in text or "mall" in text or "商场" in text) else "unspecified_building_type",
                "hazards": list(dict.fromkeys([r for r in ["high_smoke_detected" if ("smoke" in text or "烟" in text) else "", "power_status_unknown" if ("unknown" in text and "power" in text) else ""] if r])),
                "affected_people": ["public_occupants"] if ("mall" in text or "crowd" in text or "商场" in text) else [],
                "resources_or_equipment": ["respiratory_protection"] if ("smoke" in text or "烟" in text) else [],
                "emergency_stage": "initial_response",
                "information_gaps": list(dict.fromkeys(missing)),
            }
            return json.dumps(payload, ensure_ascii=False)

        # E-KELL prompt-chain stage schemas.
        if "stage 1" in user.lower() and "situation understanding" in user.lower():
            return json.dumps({
                "emergency_type": "electrical_fire" if ("electrical" in text or "power" in text) else "fire_emergency",
                "involved_entities": [x for x in ["electrical_fire" if "electrical" in text else "", "high_smoke" if "smoke" in text else "", "shopping_mall" if "mall" in text else ""] if x],
                "hazards": risks,
                "emergency_stage": "initial_response",
                "missing_information": list(dict.fromkeys(missing)),
                "evidence_used": citations,
            }, ensure_ascii=False)

        if "stage 2" in user.lower() and "kg-grounded decision reasoning" in user.lower():
            return json.dumps({
                "reasoning_summary": "Candidate actions are derived only from the scenario and retrieved KG/evidence supplied to this external baseline.",
                "candidate_actions": actions,
                "deferred_or_unsupported_actions": blocked,
                "missing_information": list(dict.fromkeys(missing)),
                "evidence_links": citations,
            }, ensure_ascii=False)

        gate = "critical_information_missing_or_requires_human_confirmation" if missing or blocked else "baseline_response_without_explicit_gate"
        return json.dumps({
            "situation_summary": "External baseline response generated from the scenario text and retrieved context supplied to the method.",
            "key_risks": list(dict.fromkeys(risks)),
            "recommended_actions": list(dict.fromkeys(actions)),
            "blocked_or_unsafe_actions": list(dict.fromkeys(blocked)),
            "missing_confirmations": list(dict.fromkeys(missing)),
            "supporting_evidence": evidence,
            "citations": citations,
            "final_decision_gate": gate,
        }, ensure_ascii=False)


@dataclass
class OpenAIChatClient:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    provider: str = "openai"
    heuristic_fallback: bool = False

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


def _llm_cfg(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    return dict(config.get("llm", config) if isinstance(config, dict) else {})


def build_llm_client(config: dict[str, Any] | None = None) -> LLMClient:
    llm_cfg = _llm_cfg(config)
    provider = str(llm_cfg.get("provider", "heuristic")).lower()
    model = str(llm_cfg.get("model", "local-deterministic-heuristic-smoke-test"))
    if provider in {"openai", "openai_chat", "openai-compatible", "deepseek", "qwen"}:
        return OpenAIChatClient(
            model=model,
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url_env=str(llm_cfg.get("base_url_env", "OPENAI_BASE_URL")),
            provider=provider,
        )
    return HeuristicLLMClient(model=model, provider=provider or "heuristic")


def llm_config_summary(config: dict[str, Any] | None = None, llm: Any | None = None) -> dict[str, Any]:
    llm_cfg = _llm_cfg(config)
    provider = str(llm_cfg.get("provider") or getattr(llm, "provider", "heuristic"))
    model = str(llm_cfg.get("model") or getattr(llm, "model", "unknown"))
    return {
        "provider": provider,
        "model": model,
        "temperature": float(llm_cfg.get("temperature", 0.0)),
        "max_tokens": int(llm_cfg.get("max_tokens", 1200)),
        "heuristic_fallback": bool(provider.lower() == "heuristic" or getattr(llm, "heuristic_fallback", False)),
    }
