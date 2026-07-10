from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        ...


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    def add(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += int(prompt)
        self.completion_tokens += int(completion)
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.llm_calls += 1

    def snapshot(self) -> TokenUsage:
        """Return an immutable-in-practice copy of the current counters."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            llm_calls=self.llm_calls,
        )

    def difference(self, earlier: TokenUsage) -> TokenUsage:
        """Return usage accrued since an earlier snapshot."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
            total_tokens=self.total_tokens - earlier.total_tokens,
            llm_calls=self.llm_calls - earlier.llm_calls,
        )

    delta = difference

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


@dataclass
class UsageTrackingLLMClient:
    """Wrapper that records call counts and token usage when available."""

    inner: Any
    usage: TokenUsage = field(default_factory=TokenUsage)
    last_latency_ms: float = 0.0

    @property
    def provider(self) -> str:
        return str(getattr(self.inner, "provider", "unknown"))

    @property
    def model(self) -> str:
        return str(getattr(self.inner, "model", "unknown"))

    @property
    def heuristic_fallback(self) -> bool:
        return bool(getattr(self.inner, "heuristic_fallback", False))

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Prefer richer signature when available.
        try:
            text = self.inner.complete(**kwargs, top_p=top_p, seed=seed)
        except TypeError:
            text = self.inner.complete(**kwargs)
        self.last_latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        usage = getattr(self.inner, "last_usage", None)
        if isinstance(usage, dict):
            self.usage.add(
                prompt=int(usage.get("prompt_tokens") or 0),
                completion=int(usage.get("completion_tokens") or 0),
            )
        else:
            # Approximate token counts for heuristic/smoke only (not for paper cost claims).
            approx_prompt = max(1, (len(system) + len(user)) // 4)
            approx_completion = max(1, len(text or "") // 4)
            self.usage.add(prompt=approx_prompt, completion=approx_completion)
        return text

    def usage_snapshot(self) -> TokenUsage:
        return self.usage.snapshot()

    def usage_delta(self, earlier: TokenUsage) -> TokenUsage:
        return self.usage.difference(earlier)


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
    last_usage: dict[str, int] = field(default_factory=dict)

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
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
            found += sorted(set(re.findall(r'"(?:citation|source_id|chunk_id|context_id|evidence_id|triple_id|path_id)"\s*:\s*"([^"]+)"', user)))
            citations.extend(found[:8])

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
            out = json.dumps(payload, ensure_ascii=False)
            self.last_usage = {"prompt_tokens": max(1, (len(system) + len(user)) // 4), "completion_tokens": max(1, len(out) // 4)}
            return out

        if "stage 1" in user.lower() and "situation understanding" in user.lower():
            out = json.dumps({
                "emergency_type": "electrical_fire" if ("electrical" in text or "power" in text) else "fire_emergency",
                "involved_entities": [x for x in ["electrical_fire" if "electrical" in text else "", "high_smoke" if "smoke" in text else "", "shopping_mall" if "mall" in text else ""] if x],
                "hazards": risks,
                "emergency_stage": "initial_response",
                "missing_information": list(dict.fromkeys(missing)),
                "evidence_used": citations,
            }, ensure_ascii=False)
            self.last_usage = {"prompt_tokens": max(1, (len(system) + len(user)) // 4), "completion_tokens": max(1, len(out) // 4)}
            return out

        # E-KELL logical query decomposition (constrained AST only).
        if "constrained json ast" in text or "ast shape example" in text or "known entities:" in text:
            entity = "electrical fire" if ("electrical" in text or "电" in text) else ("high smoke" if ("smoke" in text or "烟" in text) else "unknown")
            relation = "requires_confirmation" if "electrical" in text or "power" in text else ("requires" if "smoke" in text else "related_to")
            if "smoke" in text and ("electrical" in text or "power" in text):
                payload = {
                    "operation": "intersection",
                    "operands": [
                        {"operation": "projection", "entity": "electrical fire", "relation": "requires_confirmation"},
                        {"operation": "projection", "entity": "high smoke", "relation": "requires"},
                    ],
                }
            else:
                payload = {"operation": "projection", "entity": entity, "relation": relation}
            out = json.dumps(payload, ensure_ascii=False)
            self.last_usage = {"prompt_tokens": max(1, (len(system) + len(user)) // 4), "completion_tokens": max(1, len(out) // 4)}
            return out

        # Unified decision+response schema takes priority (including E-KELL final stage).
        if (
            '"decision"' in system.lower()
            or '"decision"' in user.lower()
            or "decision schema" in user.lower()
            or "return only one json object with this exact top-level shape" in user.lower()
            or "allowed risk_signals:" in user.lower()
        ):
            tax_risks: list[str] = []
            tax_actions: list[dict[str, Any]] = []
            tax_blocked: list[str] = []
            tax_missing: list[str] = []
            if "electrical" in text or "power" in text or "电" in text:
                tax_risks.extend(["electrical_risk", "power_cutoff_unknown", "fire_detected"])
                tax_actions.append(
                    {
                        "action_id": "verify_power_isolation",
                        "text": "Confirm power isolation status before selecting suppression actions.",
                        "priority": "high",
                        "evidence_refs": citations[:3],
                    }
                )
                tax_missing.append("power_cutoff_status")
                tax_blocked.append("BLOCK_UNVERIFIED_WATER_SUPPRESSION")
            if "smoke" in text or "烟" in text:
                tax_risks.extend(["smoke_detected", "high_smoke_detected"])
                tax_actions.append(
                    {
                        "action_id": "prepare_respiratory_protection",
                        "text": "Prepare respiratory protection before any interior entry.",
                        "priority": "high",
                        "evidence_refs": citations[:3],
                    }
                )
                tax_missing.append("respiratory_protection_status")
                tax_blocked.append("BLOCK_ENTRY_WITHOUT_RESPIRATORY_PROTECTION")
            if "crowd" in text or "evac" in text or "mall" in text or "商场" in text:
                tax_risks.append("people_at_risk")
                tax_actions.append(
                    {
                        "action_id": "confirm_evacuation_route",
                        "text": "Confirm evacuation route status before directing occupants.",
                        "priority": "high",
                        "evidence_refs": citations[:3],
                    }
                )
                tax_missing.append("route_status")
            if "chemical" in text or "hazmat" in text or "化学" in text:
                tax_risks.append("hazardous_material_risk")
                tax_missing.append("equipment_status")
            if not tax_risks:
                tax_risks.append("evidence_needed")
                tax_missing.append("sensor_freshness")
                tax_actions.append(
                    {
                        "action_id": "request_human_confirmation",
                        "text": "Collect missing incident details and request human confirmation.",
                        "priority": "high",
                        "evidence_refs": [],
                    }
                )
            # Deduplicate while preserving order.
            tax_risks = list(dict.fromkeys(tax_risks))
            tax_blocked = list(dict.fromkeys(tax_blocked))
            tax_missing = list(dict.fromkeys(tax_missing))
            seen_aids: set[str] = set()
            deduped_actions = []
            for a in tax_actions:
                if a["action_id"] in seen_aids:
                    continue
                seen_aids.add(a["action_id"])
                deduped_actions.append(a)
            human_review = bool(tax_missing or tax_blocked)
            gate = "await_human_confirmation" if human_review else "allow_response"
            status = "awaiting_human_confirmation" if human_review else "provided"
            nl = (
                "External baseline response generated from the scenario text and retrieved context "
                "supplied to the method. Confirm missing status before irreversible actions."
            )
            out = json.dumps(
                {
                    "decision": {
                        "risk_signals": tax_risks,
                        "risk_level": "high" if human_review else "medium",
                        "recommended_actions": deduped_actions,
                        "blocked_actions": tax_blocked,
                        "missing_confirmations": tax_missing,
                        "human_review_required": human_review,
                        "final_decision_gate": gate,
                    },
                    "response": {
                        "status": status,
                        "text": nl,
                        "citations": citations[:8],
                    },
                },
                ensure_ascii=False,
            )
            self.last_usage = {
                "prompt_tokens": max(1, (len(system) + len(user)) // 4),
                "completion_tokens": max(1, len(out) // 4),
            }
            return out

        # Stepwise FOL prompt-chain operations / final KG-grounded response (legacy paper format).
        if "entities connected to" in text or "intersection of" in text or "union of" in text or "do not belong" in text or "based on the above information" in text or "kg_context" in user.lower() or "logical_ast" in user.lower() or "step_results" in user.lower():
            out = json.dumps({
                "entities": ["power isolation", "respiratory protection"] if "smoke" in text or "electrical" in text else [],
                "summary": "KG-grounded decision support from retrieved triples and expanded neighborhood.",
                "situation_summary": "KG-grounded decision support from retrieved triples and expanded neighborhood.",
                "risks": risks,
                "key_risks": risks,
                "actions": actions,
                "recommended_actions": [{"action_id": "action_1", "text": a, "evidence_refs": citations[:2]} for a in actions[:3]],
                "blocked_or_unsafe_actions": blocked,
                "missing_information": missing,
                "missing_confirmations": missing,
                "evidence_ids": citations[:8],
                "citations": citations[:8],
                "response": "Use retrieved KG facts only; confirm missing status before irreversible actions.",
                "final_response": "Use retrieved KG facts only; confirm missing status before irreversible actions.",
                "final_decision_gate": "critical_information_missing_or_requires_human_confirmation" if missing else "baseline_response_without_explicit_gate",
            }, ensure_ascii=False)
            self.last_usage = {"prompt_tokens": max(1, (len(system) + len(user)) // 4), "completion_tokens": max(1, len(out) // 4)}
            return out

        if "stage 2" in user.lower() and "kg-grounded decision reasoning" in user.lower():
            out = json.dumps({
                "reasoning_summary": "Candidate actions are derived only from the scenario and retrieved KG/evidence supplied to this external baseline.",
                "candidate_actions": actions,
                "deferred_or_unsupported_actions": blocked,
                "missing_information": list(dict.fromkeys(missing)),
                "evidence_links": citations,
            }, ensure_ascii=False)
            self.last_usage = {"prompt_tokens": max(1, (len(system) + len(user)) // 4), "completion_tokens": max(1, len(out) // 4)}
            return out

        legacy_gate = (
            "critical_information_missing_or_requires_human_confirmation"
            if missing or blocked
            else "baseline_response_without_explicit_gate"
        )
        out = json.dumps(
            {
                "situation_summary": "External baseline response generated from the scenario text and retrieved context supplied to the method.",
                "key_risks": list(dict.fromkeys(risks)),
                "recommended_actions": list(dict.fromkeys(actions)),
                "blocked_or_unsafe_actions": list(dict.fromkeys(blocked)),
                "missing_confirmations": list(dict.fromkeys(missing)),
                "supporting_evidence": evidence,
                "citations": citations,
                "final_decision_gate": legacy_gate,
            },
            ensure_ascii=False,
        )
        self.last_usage = {"prompt_tokens": max(1, (len(system) + len(user)) // 4), "completion_tokens": max(1, len(out) // 4)}
        return out


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def _maybe_load_dotenv() -> None:
    """Deprecated wrapper — use load_local_environment()."""
    from external_baselines.common.environment import load_local_environment

    load_local_environment()


def resolve_siliconflow_model(llm_cfg: dict[str, Any], *, paper_final: bool = False) -> tuple[str, str]:
    """Resolve SiliconFlow model. YAML is the formal authority.

    Returns (model, model_source).
    """
    yaml_model = str(llm_cfg.get("model") or "").strip()
    allow_override = bool(llm_cfg.get("allow_model_env_override", False))
    env_model = str(os.getenv("SILICONFLOW_MODEL") or "").strip()

    if paper_final and allow_override:
        raise ValueError(
            "paper_final=true forbids llm.allow_model_env_override=true. "
            "Formal model identity must come from YAML."
        )

    if allow_override and not paper_final and env_model:
        return env_model, "env_override"
    if yaml_model:
        return yaml_model, "yaml_config"
    return DEFAULT_SILICONFLOW_MODEL, "default_constant"


def resolve_api_key(llm_cfg: dict[str, Any]) -> tuple[str, str]:
    """Resolve API key from configured env names. Never logs the key value."""
    candidates = [
        str(llm_cfg.get("api_key_env") or ""),
        "SILICONFLOW_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
    ]
    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        value = os.getenv(name)
        if value and value.strip():
            key = value.strip()
            try:
                key.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"{name} contains non-ASCII characters. Use the real API key, not a placeholder."
                ) from exc
            return key, name
    raise RuntimeError(
        "No LLM API key found. Set SILICONFLOW_API_KEY (preferred, matches fire-agent-demo) "
        "or LLM_API_KEY / OPENAI_API_KEY."
    )


@dataclass
class OpenAIChatClient:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    provider: str = "openai"
    heuristic_fallback: bool = False
    timeout_sec: float = 60.0
    connect_timeout_sec: float | None = None
    read_timeout_sec: float | None = None
    write_timeout_sec: float | None = None
    max_retries: int = 2
    model_version: str | None = None
    default_base_url: str | None = None
    enable_thinking: bool | None = None
    last_usage: dict[str, int] = field(default_factory=dict)
    api_key_env_used: str | None = None

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed. Use heuristic provider or install openai.") from exc
        api_key, used_env = resolve_api_key({"api_key_env": self.api_key_env})
        self.api_key_env_used = used_env
        base_url = os.getenv(self.base_url_env) or self.default_base_url
        timeout = self.timeout_sec
        if self.read_timeout_sec is not None:
            timeout = float(self.read_timeout_sec)
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": self.max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            create_kwargs["top_p"] = top_p
        if seed is not None:
            create_kwargs["seed"] = seed
        if self.enable_thinking is not None:
            create_kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        response = client.chat.completions.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        else:
            self.last_usage = {}
        return response.choices[0].message.content or ""


def _llm_cfg(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    return dict(config.get("llm", config) if isinstance(config, dict) else {})


def build_llm_client(config: dict[str, Any] | None = None, *, track_usage: bool = True) -> LLMClient:
    from external_baselines.common.environment import load_local_environment

    load_local_environment()
    llm_cfg = _llm_cfg(config)
    provider = str(llm_cfg.get("provider", "heuristic")).lower()
    model = str(llm_cfg.get("model", "local-deterministic-heuristic-smoke-test"))
    paper_final = bool((config or {}).get("paper_final", False))
    model_source = "yaml_config"

    # Align with fire-agent-demo SiliconFlow OpenAI-compatible client.
    if provider in {"siliconflow", "silicon-flow"}:
        model, model_source = resolve_siliconflow_model(llm_cfg, paper_final=paper_final)
        inner: Any = OpenAIChatClient(
            model=model,
            api_key_env=str(llm_cfg.get("api_key_env", "SILICONFLOW_API_KEY")),
            base_url_env=str(llm_cfg.get("base_url_env", "SILICONFLOW_BASE_URL")),
            provider="siliconflow",
            timeout_sec=float(llm_cfg.get("timeout_sec", llm_cfg.get("read_timeout_sec", 120.0))),
            connect_timeout_sec=float(llm_cfg["connect_timeout_sec"]) if llm_cfg.get("connect_timeout_sec") is not None else None,
            read_timeout_sec=float(llm_cfg["read_timeout_sec"]) if llm_cfg.get("read_timeout_sec") is not None else None,
            write_timeout_sec=float(llm_cfg["write_timeout_sec"]) if llm_cfg.get("write_timeout_sec") is not None else None,
            max_retries=int(llm_cfg.get("max_retries", 0)),
            model_version=str(llm_cfg.get("model_version") or llm_cfg.get("version") or model),
            default_base_url=DEFAULT_SILICONFLOW_BASE_URL,
            enable_thinking=(
                bool(llm_cfg.get("enable_thinking"))
                if llm_cfg.get("enable_thinking") is not None
                else None
            ),
        )
        setattr(inner, "model_source", model_source)
    elif provider in {"openai", "openai_chat", "openai-compatible", "deepseek", "qwen"}:
        inner = OpenAIChatClient(
            model=model,
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url_env=str(llm_cfg.get("base_url_env", "OPENAI_BASE_URL")),
            provider=provider,
            timeout_sec=float(llm_cfg.get("timeout_sec", 60.0)),
            max_retries=int(llm_cfg.get("max_retries", 2)),
            model_version=str(llm_cfg.get("model_version") or llm_cfg.get("version") or "") or None,
        )
        setattr(inner, "model_source", "yaml_config")
    else:
        inner = HeuristicLLMClient(model=model, provider=provider or "heuristic")
        setattr(inner, "model_source", "yaml_config")
    if track_usage:
        client = UsageTrackingLLMClient(inner=inner)
        setattr(client, "model_source", model_source)
        return client
    return inner


def llm_runtime_snapshot(llm: Any | None = None) -> dict[str, Any]:
    if isinstance(llm, UsageTrackingLLMClient):
        return {
            "llm_calls": llm.usage.llm_calls,
            "token_usage": llm.usage.to_dict(),
            "last_latency_ms": llm.last_latency_ms,
            "cost": None,
        }
    return {"llm_calls": 0, "token_usage": {}, "last_latency_ms": 0.0, "cost": None}


def llm_config_summary(config: dict[str, Any] | None = None, llm: Any | None = None) -> dict[str, Any]:
    llm_cfg = _llm_cfg(config)
    provider = str(llm_cfg.get("provider") or getattr(llm, "provider", "heuristic"))
    paper_final = bool((config or {}).get("paper_final", False))
    model_source = "yaml_config"
    if provider.lower() in {"siliconflow", "silicon-flow"}:
        model, model_source = resolve_siliconflow_model(llm_cfg, paper_final=paper_final)
    else:
        model = str(llm_cfg.get("model") or getattr(llm, "model", "unknown"))
        model_source = str(getattr(llm, "model_source", None) or getattr(getattr(llm, "inner", None), "model_source", None) or "yaml_config")
    if llm is not None:
        model = str(getattr(llm, "model", None) or getattr(getattr(llm, "inner", None), "model", None) or model)
        model_source = str(
            getattr(llm, "model_source", None)
            or getattr(getattr(llm, "inner", None), "model_source", None)
            or model_source
        )
    inner = getattr(llm, "inner", llm)
    return {
        "provider": provider,
        "model": model,
        "model_version": llm_cfg.get("model_version") or llm_cfg.get("version") or getattr(inner, "model_version", None) or model,
        "model_source": model_source,
        "temperature": float(llm_cfg.get("temperature", 0.0)),
        "top_p": llm_cfg.get("top_p"),
        "max_tokens": int(llm_cfg.get("max_tokens", 1200)),
        "seed": llm_cfg.get("seed"),
        "timeout_sec": llm_cfg.get("timeout_sec") or llm_cfg.get("read_timeout_sec"),
        "max_retries": llm_cfg.get("max_retries"),
        "api_key_env": llm_cfg.get("api_key_env") or getattr(inner, "api_key_env", None),
        "base_url_env": llm_cfg.get("base_url_env") or getattr(inner, "base_url_env", None),
        "heuristic_fallback": bool(provider.lower() == "heuristic" or getattr(llm, "heuristic_fallback", False)),
    }
