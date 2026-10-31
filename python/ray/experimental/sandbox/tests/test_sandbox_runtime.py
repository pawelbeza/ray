"""SandboxRuntime-layer tests with fake manager/backend (no runsc needed).

These pin that ``create()`` arguments actually reach the backend's
``SandboxConfig`` — the layer where a declared-but-unforwarded field once
shipped behind a green suite.
"""

import asyncio
import os
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from ray.experimental.sandbox.backend.base import ExecResult, SandboxStatus
from ray.experimental.sandbox.runtime import SandboxRuntime


class _FakeImageManager:
    def pull_image(self, image, timeout_seconds=120.0):
        return f"/tmp/fake-images/{image}"


class _FakeBackend:
    def __init__(self):
        self.configs = []
        self.deleted = []

    def create_sandbox(self, config):
        self.configs.append(config)
        return "ray-sandbox-fake0001"

    def delete_sandbox(self, sandbox_id):
        self.deleted.append(sandbox_id)

    def exec_command(
        self,
        sandbox_id,
        command,
        timeout=None,
        cwd=None,
        env=None,
        shell=None,
        user=None,
        combine_output=False,
    ):
        self.exec_options = {
            "timeout": timeout,
            "cwd": cwd,
            "env": env,
            "shell": shell,
            "user": user,
            "combine_output": combine_output,
        }
        return ExecResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
            output=b"\xff" if combine_output else None,
        )

    def write_file(self, sandbox_id, path, content):
        pass

    def read_file(self, sandbox_id, path):
        return b""

    def get_status(self, sandbox_id):
        return SandboxStatus.RUNNING

    def checkpoint_sandbox(
        self,
        sandbox_id: str,
        checkpoint_path=None,
        leave_running: bool = True,
        **kwargs,
    ):
        path = checkpoint_path or f"/tmp/checkpoints/{sandbox_id}"
        self.checkpoints.append(
            {
                "sandbox_id": sandbox_id,
                "checkpoint_path": path,
                "leave_running": leave_running,
                "kwargs": kwargs,
            }
        )
        return {
            "checkpoint_path": path,
            "sandbox_id": sandbox_id,
            "leave_running": leave_running,
        }

    def restore_sandbox(self, checkpoint_path: str, **kwargs):
        new_id = f"ray-sandbox-restored-{len(self.restores) + 1}"
        self.restores.append(
            {"checkpoint_path": checkpoint_path, "new_id": new_id, "kwargs": kwargs}
        )
        return new_id

    def pause_sandbox(self, sandbox_id: str, timeout_seconds=None):
        self.paused.append(sandbox_id)

    def resume_sandbox(self, sandbox_id: str, timeout_seconds=None):
        self.resumed.append(sandbox_id)


def _fake_runtime():
    runtime = SandboxRuntime()
    runtime._image_manager = _FakeImageManager()
    backend = _FakeBackend()
    backend.checkpoints = []
    backend.restores = []
    backend.paused = []
    backend.resumed = []
    runtime._backend = backend
    return runtime


def test_create_forwards_config_fields_to_backend():
    runtime = _fake_runtime()
    runtime.create(
        image="fake:latest",
        cpu=2.0,
        memory="1Gi",
        env={"A": "1"},
        workdir="/app",
        network="host",
        dns=["10.0.0.2"],
        capabilities=["CAP_CHOWN"],
        rootless=True,
        readonly=False,
        shell="/bin/bash",
        oom_score_adj=0,
    )
    (config,) = runtime._backend.configs
    assert config.image == "fake:latest"
    assert config.cpu == 2.0
    assert config.env == {"A": "1"}
    assert config.workdir == "/app"
    assert config.network == "host"
    assert config.dns == ["10.0.0.2"]
    assert config.capabilities == ["CAP_CHOWN"]
    assert config.rootless is True
    assert config.readonly is False
    assert config.shell == "/bin/bash"
    assert config.oom_score_adj == 0


@pytest.mark.parametrize("combine_output", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_exec_forwards_options_to_backend(combine_output: bool, asynchronous: bool):
    runtime = _fake_runtime()
    options = {
        "timeout": 5,
        "cwd": "/workspace",
        "env": {"A": "1"},
        "shell": "/bin/sh",
        "user": "1000",
        "combine_output": combine_output,
    }
    if asynchronous:
        result = asyncio.run(runtime.exec_async("sandbox", "true", **options))
    else:
        result = runtime.exec("sandbox", "true", **options)
    assert runtime._backend.exec_options == options
    assert result.output == (b"\xff" if combine_output else None)


def _fake_running_backend(tmp_path):
    from ray.experimental.sandbox.backend.gvisor import GVisorSandboxBackend
    from ray.experimental.sandbox.config import SandboxConfig

    backend = GVisorSandboxBackend(image_manager=_FakeImageManager())
    backend._sandbox_metadata["sandbox"] = {
        "config": SandboxConfig(image="fake:latest"),
        "root_dir": str(tmp_path),
        "cwd": "/",
        "status": SandboxStatus.RUNNING,
    }
    return backend


@pytest.mark.parametrize("combine_output", [False, True])
def test_gvisor_exec_binary_output(tmp_path, monkeypatch, combine_output: bool):
    backend = _fake_running_backend(tmp_path)
    popen = subprocess.Popen

    def local_popen(args, **kwargs):
        return popen(
            [
                sys.executable,
                "-c",
                "import os; os.write(1, b'out\\xff'); "
                "os.write(2, b'err\\xfe'); os.write(1, b'end')",
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", local_popen)
    result = backend.exec_command("sandbox", ["true"], combine_output=combine_output)
    assert result.exit_code == 0
    if combine_output:
        assert result.output == b"out\xfferr\xfeend"
        assert result.stdout == result.stderr == ""
    else:
        assert result.output is None
        assert result.stdout == "out\ufffdend"
        assert result.stderr == "err\ufffd"


@pytest.mark.parametrize("combine_output", [False, True])
def test_gvisor_exec_timeout_closes_available_pipes(
    tmp_path, monkeypatch, combine_output: bool
):
    from ray.experimental.sandbox.exceptions import SandboxTimeoutError

    backend = _fake_running_backend(tmp_path)
    proc = Mock()
    proc.stderr = None if combine_output else Mock()
    proc.communicate.side_effect = subprocess.TimeoutExpired("runsc", 1)
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=proc))
    kill_exec = Mock()
    monkeypatch.setattr(backend, "_kill_exec", kill_exec)
    with pytest.raises(SandboxTimeoutError):
        backend.exec_command(
            "sandbox", ["true"], timeout=1, combine_output=combine_output
        )
    kill_exec.assert_called_once()
    proc.kill.assert_called_once()
    proc.stdout.close.assert_called_once()
    if not combine_output:
        proc.stderr.close.assert_called_once()
    proc.wait.assert_called_once()


@pytest.mark.parametrize(
    "append,returncodes",
    [(False, [0]), (False, [127, 0]), (True, [0]), (False, [1, 1])],
)
def test_gvisor_write_fast_path_and_fallback(
    tmp_path, monkeypatch, append: bool, returncodes: list
):
    from ray.experimental.sandbox.exceptions import SandboxError

    backend = _fake_running_backend(tmp_path)
    commands = []
    processes = []

    def fake_popen(args, **kwargs):
        commands.append(args)
        proc = Mock(returncode=returncodes[len(processes)])
        proc.communicate.return_value = (b"", b"write denied")
        processes.append(proc)
        return proc

    wrap = Mock(side_effect=lambda meta, args: ["wrapped", *args])
    monkeypatch.setattr(backend, "_wrap_workload", wrap)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    path = "/workspace/nested/file with spaces"
    if returncodes[-1] != 0:
        with pytest.raises(SandboxError, match="write denied"):
            backend.write_file("sandbox", path, "caf\u00e9", append=append)
    else:
        backend.write_file("sandbox", path, "caf\u00e9", append=append)
    assert len(commands) == len(returncodes)
    assert wrap.call_count == len(commands)
    for command, proc in zip(commands, processes):
        assert command[0] == "wrapped"
        assert command[-1] == path
        proc.communicate.assert_called_once_with(input=b"caf\xc3\xa9")
    if not append:
        assert commands[0][-6:] == ["install", "-D", "-m", "0644", "/dev/stdin", path]
    if append or len(commands) > 1:
        assert "/bin/sh" in commands[-1]
        assert ("cat >>" if append else "cat >") in commands[-1][-3]


def test_ttl_is_enforced_by_the_runtime():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest", ttl_seconds=1)
    assert runtime._backend.deleted == []
    deadline = time.monotonic() + 5
    while not runtime._backend.deleted and time.monotonic() < deadline:
        time.sleep(0.05)
    assert runtime._backend.deleted == [instance_id]
    # The timer is gone after firing (delete popped it).
    assert instance_id not in runtime._ttl_timers


def test_delete_cancels_the_ttl_timer():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest", ttl_seconds=3600)
    timer = runtime._ttl_timers[instance_id]
    runtime.delete(instance_id)
    assert instance_id not in runtime._ttl_timers
    # cancel() sets the finished event; the thread itself may take a beat to
    # exit, so assert the deterministic signal rather than thread liveness.
    assert timer.finished.is_set()
    assert runtime._backend.deleted == [instance_id]


def test_no_ttl_by_default():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest")
    assert instance_id not in runtime._ttl_timers
    (config,) = runtime._backend.configs
    assert config.ttl_seconds is None
    assert config.shell == "/bin/bash"


def test_checkpoint_and_restore_forward_to_backend():
    runtime = _fake_runtime()
    instance_id = runtime.create(image="fake:latest")

    # Checkpoint with leave_running=True
    ckpt_path = runtime.checkpoint(
        instance_id,
        checkpoint_path="/tmp/test_checkpoint",
        leave_running=True,
    )
    assert ckpt_path == "/tmp/test_checkpoint"
    assert len(runtime._backend.checkpoints) == 1
    assert runtime._backend.checkpoints[0]["sandbox_id"] == instance_id
    assert runtime._backend.checkpoints[0]["checkpoint_path"] == "/tmp/test_checkpoint"
    assert runtime._backend.checkpoints[0]["leave_running"] is True

    # Restore from checkpoint
    restored_id = runtime.restore(
        checkpoint_path=ckpt_path,
        cpu=4.0,
        memory="2Gi",
    )
    assert restored_id.startswith("ray-sandbox-restored-")
    assert len(runtime._backend.restores) == 1
    assert runtime._backend.restores[0]["checkpoint_path"] == ckpt_path
    assert runtime._backend.restores[0]["kwargs"]["cpu"] == 4.0
    assert runtime._backend.restores[0]["kwargs"]["memory"] == "2Gi"


def test_gvisor_backend_pause_resume_lifecycle(tmp_path, monkeypatch):
    """Test GVisorSandboxBackend pause/resume handling, state transitions, and error paths."""
    from ray.experimental.sandbox.backend.gvisor import GVisorSandboxBackend
    from ray.experimental.sandbox.config import SandboxConfig
    from ray.experimental.sandbox.exceptions import SandboxError

    backend = GVisorSandboxBackend(image_manager=_FakeImageManager())
    config = SandboxConfig(image="fake:latest")
    sandbox_id = "test-sandbox-id"
    root_dir = str(tmp_path / sandbox_id)
    os.makedirs(root_dir, exist_ok=True)
    backend._sandbox_metadata[sandbox_id] = {
        "config": config,
        "root_dir": root_dir,
        "status": SandboxStatus.RUNNING,
    }

    commands_executed = []

    def fake_run(args, **kwargs):
        commands_executed.append(args)

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)

    # 1. Normal pause
    backend.pause_sandbox(sandbox_id)
    assert backend.get_status(sandbox_id) == SandboxStatus.PAUSED
    assert any("pause" in cmd for cmd in commands_executed)

    # 2. Idempotent pause (should not re-run runsc)
    len_before = len(commands_executed)
    backend.pause_sandbox(sandbox_id)
    assert len(commands_executed) == len_before

    # 3. Normal resume
    backend.resume_sandbox(sandbox_id)
    assert backend.get_status(sandbox_id) == SandboxStatus.RUNNING
    assert any("resume" in cmd for cmd in commands_executed)

    # 4. Idempotent resume (should not re-run runsc)
    len_before = len(commands_executed)
    backend.resume_sandbox(sandbox_id)
    assert len(commands_executed) == len_before

    # 5. Terminated sandbox error
    backend._sandbox_metadata[sandbox_id]["status"] = SandboxStatus.TERMINATED
    with pytest.raises(SandboxError, match="stopped unexpectedly"):
        backend.pause_sandbox(sandbox_id)
    with pytest.raises(SandboxError, match="stopped unexpectedly"):
        backend.resume_sandbox(sandbox_id)

    # 6. runsc error raises SandboxError
    backend._sandbox_metadata[sandbox_id]["status"] = SandboxStatus.RUNNING

    def fake_failing_run(args, **kwargs):
        class Failed:
            returncode = 1
            stdout = ""
            stderr = "runsc execution error"

        return Failed()

    monkeypatch.setattr(subprocess, "run", fake_failing_run)
    with pytest.raises(SandboxError, match="Failed to pause"):
        backend.pause_sandbox(sandbox_id)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
