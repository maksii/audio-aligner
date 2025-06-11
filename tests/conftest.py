import subprocess
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def _mock_subprocess_run(monkeypatch):
    """Stub out *subprocess.run* so that no external CLI tools are executed
    during the test-suite. This keeps the tests hermetic and fast.
    """

    mocked = Mock(return_value=Mock(returncode=0, stdout=b"", stderr=b""))
    monkeypatch.setattr(subprocess, "run", mocked) 