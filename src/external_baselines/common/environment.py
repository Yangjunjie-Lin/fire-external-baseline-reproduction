"""Shared local environment loading (credentials only; never expose secrets)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PRESENCE_VARS = (
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "SILICONFLOW_MODEL",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
)


def _try_load_dotenv(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except Exception:
        return False
    load_dotenv(path, override=False)
    return True


def load_local_environment(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Load env files with override=False. Never returns secret values.

    Priority (later files do not override earlier process env or earlier files):
    1. Existing process environment (preserved)
    2. baseline repo/.env
    3. FIRE_AGENT_DEMO_ENV_PATH if set
    4. sibling ../fire-agent-demo/.env (only if FIRE_AGENT_DEMO_ENV_PATH unset)
    """
    root = Path(repo_root) if repo_root is not None else ROOT
    loaded: list[str] = []
    discovered: list[str] = []

    baseline_env = root / ".env"
    if baseline_env.is_file():
        discovered.append("baseline_repo_env")
        if _try_load_dotenv(baseline_env):
            loaded.append("baseline_repo_env")

    demo_env = os.getenv("FIRE_AGENT_DEMO_ENV_PATH")
    if demo_env:
        explicit = Path(demo_env)
        if explicit.is_file():
            discovered.append("explicit_main_env_path")
            if _try_load_dotenv(explicit):
                loaded.append("explicit_main_env_path")
    else:
        sibling = root.parent / "fire-agent-demo" / ".env"
        if sibling.is_file():
            discovered.append("sibling_main_project_env")
            if _try_load_dotenv(sibling):
                loaded.append("sibling_main_project_env")

    return {
        "loaded_sources": loaded,
        "discovered_sources": discovered,
        "override": False,
    }


def environment_variable_presence(names: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Return present/missing labels only — never values."""
    names = list(names) if names is not None else list(DEFAULT_PRESENCE_VARS)
    return {name: ("present" if os.getenv(name) else "missing") for name in names}
