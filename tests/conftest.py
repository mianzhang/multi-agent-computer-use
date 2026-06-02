from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def reset_manager_prompt_counter(monkeypatch):
    import run_macu
    import utils.manager_utils

    monkeypatch.setattr(run_macu, "_manager_call_counter", 0, raising=False)
    monkeypatch.setattr(utils.manager_utils, "_manager_call_counter", 0, raising=False)
