from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.quality import data_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate World Cup prediction data and model artifacts.")
    parser.add_argument("--output", default="", help="Optional JSON output path for the validation report.")
    args = parser.parse_args()

    report = data_quality_report()
    payload = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)
    print(payload)
    return 0 if report["status"] in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
