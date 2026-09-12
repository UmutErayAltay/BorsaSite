#!/usr/bin/env python3
"""Sanal alım-satım motorunu tek başına çalıştır — python scripts/run_trading.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.db import get_connection, init_schema  # noqa: E402
from trading.config import load_trading_config  # noqa: E402
from trading.engine import run_once  # noqa: E402

load_project_env()


def main() -> None:
    cfg = load_trading_config()
    with get_connection() as conn:
        init_schema(conn)
        result = run_once(conn, cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
