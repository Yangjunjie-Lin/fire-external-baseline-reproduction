#!/usr/bin/env python3
"""Alias entry for heuristic interop smoke (no paid API).

Delegates to scripts/smoke_main_runner_bundle.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "smoke_main_runner_bundle.py"
    runpy.run_path(str(target), run_name="__main__")
