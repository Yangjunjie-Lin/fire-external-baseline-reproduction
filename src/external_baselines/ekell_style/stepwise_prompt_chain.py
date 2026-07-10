from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from external_baselines.common.checksums import prompt_hash, sha256_text
from external_baselines.common.llm_client import LLMClient
from external_baselines.common.text_utils import extract_json_object
from external_baselines.ekell_style.logical_query.parser import parse_query
from external_baselines.ekell_style.logical_query.schema import QueryNode, ValidationResult

DEFAULT_PROMPT_DIR = Path("configs/prompts/paper_fidelity")
OPERATION_FILES = {
    "projection": "stepwise_projection.txt",
    "intersection": "stepwise_intersection.txt",
    "union": "stepwise_union.txt",
    "negation": "stepwise_negation.txt",
}
FINAL_PROMPT_FILE = "final_kg_grounded_response.txt"
SYSTEM_PROMPT = (
    "Execute the validated logical operation over only the supplied KG evidence. "
    "Return valid JSON. Never invent entity, triple, path, source, chunk, or evidence IDs."
)
ID_KEYS = frozenset({
    "id", "entity_id", "triple_id", "path_id", "evidence_id", "context_id",
    "source_id", "chunk_id",
})
CITATION_KEYS = frozenset({"evidence_ids", "citations", "evidence_links", "evidence_refs"})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _collect_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in ID_KEYS and child not in (None, ""):
                if isinstance(child, (list, tuple, set)):
                    found.update(str(item) for item in child)
                else:
                    found.add(str(child))
            found.update(_collect_ids(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            found.update(_collect_ids(child))
    return found


def _filter_citations(value: Any, allowed_ids: set[str]) -> tuple[Any, list[str]]:
    removed: list[str] = []
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if key in CITATION_KEYS:
                values = child if isinstance(child, list) else [child]
                kept = [str(item) for item in values if str(item) in allowed_ids]
                removed.extend(str(item) for item in values if str(item) not in allowed_ids)
                output[key] = kept
            else:
                output[key], child_removed = _filter_citations(child, allowed_ids)
                removed.extend(child_removed)
        return output, removed
    if isinstance(value, list):
        output_list = []
        for child in value:
            filtered, child_removed = _filter_citations(child, allowed_ids)
            output_list.append(filtered)
            removed.extend(child_removed)
        return output_list, removed
    return value, removed


def _validated_plan(value: ValidationResult | QueryNode | Mapping[str, Any]) -> QueryNode:
    if isinstance(value, ValidationResult):
        if not value.valid or value.plan is None:
            raise ValueError("stepwise prompt chain requires a valid logical AST")
        return value.plan
    if isinstance(value, QueryNode):
        return value
    return parse_query(value)


class StepwisePromptChain:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompt_dir: str | Path = DEFAULT_PROMPT_DIR,
        max_retries: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> None:
        self.llm = llm
        self.prompt_dir = Path(prompt_dir)
        self.max_retries = max(0, max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _template(self, filename: str) -> str:
        path = self.prompt_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing paper-fidelity prompt: {path}")
        return path.read_text(encoding="utf-8")

    def _call(self, rendered: str) -> tuple[str, Any, str, int]:
        raw = ""
        for retry in range(self.max_retries + 1):
            raw = self.llm.complete(
                system=SYSTEM_PROMPT,
                user=rendered,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            parsed = extract_json_object(raw)
            if isinstance(parsed, dict):
                return raw, parsed, "parsed", retry
        return raw, None, "failed", self.max_retries

    def run(
        self,
        *,
        validated_ast: ValidationResult | QueryNode | Mapping[str, Any],
        kg_contexts: Sequence[Any],
        kg_paths: Sequence[Any] | None = None,
        query: str = "",
        candidate_universe: Sequence[str] | None = None,
        fol_execution: Any | None = None,
    ) -> dict[str, Any]:
        plan = _validated_plan(validated_ast)
        evidence = {"contexts": list(kg_contexts), "paths": list(kg_paths or [])}
        allowed_ids = _collect_ids(evidence)
        context_json = _json(evidence)
        steps: list[dict[str, Any]] = []

        def execute(node: QueryNode) -> str:
            dependency_ids = [execute(operand) for operand in node.operands]
            step_id = f"step_{len(steps) + 1:04d}"
            dependencies = [
                {
                    "step_id": dependency_id,
                    "parsed_output": next(
                        step["parsed_output"] for step in steps if step["step_id"] == dependency_id
                    ),
                }
                for dependency_id in dependency_ids
            ]
            dependency_payload: dict[str, Any] = {"steps": dependencies}
            if node.operation == "negation":
                dependency_payload["candidate_universe"] = list(candidate_universe or [])
            template_file = OPERATION_FILES[node.operation]
            template = self._template(template_file)
            rendered = (
                template.replace("{entity}", node.entity or "(dependency result)")
                .replace("{relation}", node.relation or "")
                .replace("{dependency_results}", _json(dependency_payload))
                .replace("{kg_context}", context_json)
            )
            raw, parsed, status, retries = self._call(rendered)
            filtered, removed = _filter_citations(parsed, allowed_ids)
            steps.append({
                "step_id": step_id,
                "operation": node.operation,
                "step_dependencies": dependency_ids,
                "prompt_template_file": template_file,
                "prompt_template_hash": sha256_text(template),
                "rendered_prompt_hash": prompt_hash(SYSTEM_PROMPT, rendered),
                "raw_output": raw,
                "parsed_output": filtered,
                "parsing_status": status,
                "retry_count": retries,
                "removed_unprovided_evidence_ids": sorted(set(removed)),
            })
            return step_id

        root_step_id = execute(plan)
        final_template = self._template(FINAL_PROMPT_FILE)
        fol_json = _json(fol_execution) if fol_execution is not None else "(none)"
        final_rendered = (
            final_template.replace("{query}", query)
            .replace("{logical_ast}", _json(plan.to_dict()))
            .replace("{step_results}", _json(steps))
            .replace("{kg_context}", context_json)
            .replace("{fol_execution}", fol_json)
        )
        from external_baselines.common.decision_output import decision_schema_instruction

        final_rendered = f"{final_rendered.rstrip()}\n\n{decision_schema_instruction()}"
        raw, parsed, status, retries = self._call(final_rendered)
        final_output, removed = _filter_citations(parsed, allowed_ids)
        final_trace = {
            "step_id": "final",
            "operation": "final_kg_grounded_response",
            "step_dependencies": [root_step_id],
            "prompt_template_file": FINAL_PROMPT_FILE,
            "prompt_template_hash": sha256_text(final_template),
            "rendered_prompt_hash": prompt_hash(SYSTEM_PROMPT, final_rendered),
            "raw_output": raw,
            "parsed_output": final_output,
            "parsing_status": status,
            "retry_count": retries,
            "removed_unprovided_evidence_ids": sorted(set(removed)),
        }
        return {
            "validated_ast": plan.to_dict(),
            "allowed_evidence_ids": sorted(allowed_ids),
            "steps": steps,
            "final": final_trace,
            "final_response": final_output,
        }


def run_stepwise_prompt_chain(
    *,
    validated_ast: ValidationResult | QueryNode | Mapping[str, Any],
    kg_contexts: Sequence[Any],
    llm: LLMClient,
    kg_paths: Sequence[Any] | None = None,
    query: str = "",
    candidate_universe: Sequence[str] | None = None,
    prompt_dir: str | Path = DEFAULT_PROMPT_DIR,
    max_retries: int = 1,
    fol_execution: Any | None = None,
) -> dict[str, Any]:
    return StepwisePromptChain(
        llm=llm, prompt_dir=prompt_dir, max_retries=max_retries,
    ).run(
        validated_ast=validated_ast,
        kg_contexts=kg_contexts,
        kg_paths=kg_paths,
        query=query,
        candidate_universe=candidate_universe,
        fol_execution=fol_execution,
    )


execute_stepwise_prompt_chain = run_stepwise_prompt_chain
