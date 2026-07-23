"""Tests for batch module interrupt handling and llama-server lifecycle."""

import signal
import subprocess
import time
import urllib.request

import pytest

from frame_prep.batch import _worker_ignore_sigint, _terminate_pool_workers
from frame_prep.detector import _start_llama_server, _stop_llama_server


def test_worker_ignore_sigint():
    """Worker initializer must ignore SIGINT (and only SIGINT)."""
    original = signal.getsignal(signal.SIGINT)
    try:
        _worker_ignore_sigint()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, original)


def test_stop_llama_server_none_is_noop():
    _stop_llama_server(None)


def test_stop_llama_server_terminates():
    """A cooperative process is terminated and reaped."""
    proc = subprocess.Popen(["sleep", "60"])
    _stop_llama_server(proc)
    assert proc.poll() is not None


def test_stop_llama_server_kills_stubborn():
    """A process ignoring SIGTERM is escalated to SIGKILL."""
    proc = subprocess.Popen(["bash", "-c", "trap '' TERM; sleep 60"])
    # Give bash a moment to install the trap
    time.sleep(0.3)
    _stop_llama_server(proc, timeout=0.5)
    assert proc.poll() is not None


def test_start_llama_server_uses_new_session(monkeypatch):
    """llama-server must be detached from the terminal's process group."""
    recorded = {}

    class FakeProc:
        def terminate(self):
            pass

        def kill(self):
            pass

    def fake_popen(cmd, **kwargs):
        recorded.update(kwargs)
        return FakeProc()

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    port, proc = _start_llama_server("model.gguf", "mmproj.gguf")
    assert recorded.get("start_new_session") is True
    assert isinstance(proc, FakeProc)


def test_terminate_pool_workers_fallback():
    """Fallback path terminates processes from executor._processes."""

    class FakeWorker:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    class FakeExecutor:
        def __init__(self):
            self._processes = {1: FakeWorker(), 2: FakeWorker()}

    executor = FakeExecutor()
    _terminate_pool_workers(executor)
    assert all(w.terminated for w in executor._processes.values())
