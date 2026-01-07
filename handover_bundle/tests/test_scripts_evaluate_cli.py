from __future__ import annotations

import os
import subprocess
import sys


def test_scripts_evaluate_exits_when_no_key() -> None:
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = ""  # ensure load_dotenv can't override
    p = subprocess.run(
        [sys.executable, "-m", "scripts.evaluate", "https://example.com"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2
    assert "Missing OPENAI_API_KEY" in (p.stderr + p.stdout)


