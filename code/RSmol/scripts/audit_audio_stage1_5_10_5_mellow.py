#!/usr/bin/env python3
"""Compatibility entry point for the isolated ReasonAQA Stage 1 audit.

The delegated CLI accepts ``--report-path`` and emits ``traceback`` and
``hard_failures`` fields in its JSON report.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_reasonaqa_manifest_5_10_5_mellow import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
