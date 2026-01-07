from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run pytest with coverage reporting (requires pytest-cov)."""
    fail_under = os.environ.get("MANUAV_COV_FAIL_UNDER", "70")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--cov=manuav_eval",
        "--cov=scripts",
        "--cov-report=term-missing",
        f"--cov-fail-under={fail_under}",
    ]
    p = subprocess.run(cmd)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())


