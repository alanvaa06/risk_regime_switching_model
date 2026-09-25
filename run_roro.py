"""RoRo one-click update.

How to use:
  1. Edit the PARAMETERS block below (Notepad is fine).
  2. Double-click run_roro.bat (runs this file with the project's .venv).

Results land in outputs/historic/<CONFIG>/results_<last data date>/
(CSV files + report.html). With TYPE_RUN = "new_data" only dates not processed
yet are computed; the folder still holds the complete history.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from roro.fred_client import FredApiClient
from roro.historic import (
    TypeRun,
    available_configs,
    friendly_error,
    parse_user_date,
    run_update,
)

# ============================ PARAMETERS ============================
CONFIG = "jm-eval"  # default | eval | jm-only | jm-eval | attribution-history
TYPE_RUN = "new_data"  # "new_data" = only unprocessed dates | "all" = full history
DATA_UNTIL: str | None = None  # None = latest in data.xlsx, or "25-09-2026" / "2026-09-25"
BUILD_REPORT = True  # also write report.html into the results folder
# ====================================================================

REPO = Path(__file__).resolve().parent


def main() -> int:
    """Validate the parameters, then run. Exit code 0 ok, 1 run failed, 2 bad parameters."""
    os.chdir(REPO)  # configs use repo-relative paths (data.xlsx, outputs/)
    load_dotenv(REPO / ".env")
    configs = available_configs(REPO / "configs")
    if CONFIG not in configs:
        print(f"[x] CONFIG={CONFIG!r} not found. Valid: {', '.join(configs)}")
        return 2
    try:
        type_run = TypeRun(TYPE_RUN)
    except ValueError:
        print("[x] TYPE_RUN must be 'new_data' or 'all'")
        return 2
    try:
        until = parse_user_date(DATA_UNTIL) if DATA_UNTIL else None
    except ValueError as exc:
        print(f"[x] {exc}")
        return 2
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("[x] Add FRED_API_KEY to .env (see .env.example)")
        return 2
    try:
        run_update(
            configs[CONFIG],
            type_run=type_run,
            data_until=until,
            build_report=BUILD_REPORT,
            fred_client=FredApiClient(api_key=api_key),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort handler of a double-click runner
        message = friendly_error(exc)
        if message is None:
            traceback.print_exc()
        else:
            print(f"[x] {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
