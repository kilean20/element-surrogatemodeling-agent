"""External card helpers retain logs and cannot hang on inherited output pipes."""

import json
import subprocess

import psutil
import pytest

from element_surrogate.cards import _command


def test_nonzero_helper_preserves_combined_diagnostics(tmp_path):
    output = tmp_path / "helper.log"
    result = _command(["-c", "import sys; print('manufactured error'); sys.exit(7)"], output)
    assert result.returncode == 7
    assert "manufactured error" in output.read_text()
    assert result.stdout == output.read_text()


def test_timeout_stops_owned_descendant_and_retains_partial_log(tmp_path):
    pid_path = tmp_path / "descendant.json"
    script = tmp_path / "helper.py"
    script.write_text(
        "import json,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid}))\n"
        "print('manufactured helper started',flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    output = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        _command([str(script), str(pid_path)], output, timeout=3)
    assert "manufactured helper started" in output.read_text()
    if pid_path.exists():
        pid = json.loads(pid_path.read_text())["pid"]
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
