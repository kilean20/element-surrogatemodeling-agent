"""Numerical backends respect their explicit count and managed reservation."""

import json
import os
import subprocess
import sys

import pytest

from element_surrogate.resources import numerical_threads


def test_saved_thread_count_cannot_exceed_managed_reservation(monkeypatch):
    monkeypatch.delenv("ELEMENT_SURROGATE_NUMERICAL_THREADS", raising=False)
    assert numerical_threads(3) == 3
    monkeypatch.setenv("ELEMENT_SURROGATE_NUMERICAL_THREADS", "2")
    assert numerical_threads(2) == 2
    with pytest.raises(ValueError, match="reservation"):
        numerical_threads(3)
    with pytest.raises(ValueError, match="positive integer"):
        numerical_threads(True)


def test_managed_torch_actual_intra_and_inter_op_counts_in_fresh_process():
    pytest.importorskip("torch")
    environment = {**os.environ, "ELEMENT_SURROGATE_NUMERICAL_THREADS": "1"}
    code = """import json, torch
from element_surrogate.resources import configure_torch_threads
try:
    configure_torch_threads(torch, 2)
except ValueError:
    rejected = True
else:
    rejected = False
configure_torch_threads(torch, 1)
value = torch.ones((4,4), dtype=torch.float64) @ torch.eye(4, dtype=torch.float64)
print(json.dumps({'rejected': rejected, 'intra': torch.get_num_threads(),
                  'inter': torch.get_num_interop_threads(), 'sum': value.sum().item()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert json.loads(result.stdout) == {"rejected": True, "intra": 1, "inter": 1, "sum": 16.0}
