import sys
from unittest.mock import Mock

import pytest

from ray.experimental.sandbox._internal import slurm_steps
from ray.experimental.sandbox.backend import gvisor
from ray.experimental.sandbox.backend.base import SandboxStatus
from ray.experimental.sandbox.backend.gvisor import GVisorSandboxBackend
from ray.experimental.sandbox.config import GVisorSandboxConfig
from ray.experimental.sandbox.exceptions import SandboxError


@pytest.fixture
def slurm_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "node-a")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    monkeypatch.setenv(slurm_steps.SLURM_STEPS_ENV, "1")


def test_slurm_limits_do_not_require_delegated_cgroups(monkeypatch, slurm_job):
    backend = GVisorSandboxBackend(image_manager=Mock())
    manager = Mock(side_effect=AssertionError("Must not allocate delegated cgroups"))
    monkeypatch.setattr(gvisor, "SandboxCgroupManager", manager)
    config = GVisorSandboxConfig(image="busybox:latest", cpu=1.5, memory="3Gi")
    assert backend._cgroup_manager_for(config) is None
    assert "--ignore-cgroups" in backend._runsc_base_args(config)
    config.cpu = 0
    config.memory = None
    monkeypatch.delenv("RAY_SANDBOX_IGNORE_CGROUPS", raising=False)
    assert "--ignore-cgroups" in backend._runsc_base_args(config)
    monkeypatch.delenv(slurm_steps.SLURM_STEPS_ENV)
    assert "--ignore-cgroups" not in backend._runsc_base_args(config)


def test_slurm_oom_uses_current_termination_contract(tmp_path):
    backend = GVisorSandboxBackend(image_manager=Mock())
    log = tmp_path / "runsc.stderr.log"
    log.write_text("slurmstepd: error: Detected 1 oom_kill event\n")
    backend._sandbox_metadata["ray-sandbox-x"] = {
        "root_dir": str(tmp_path),
        "slurm_step": True,
        "stderr_log_path": str(log),
    }
    assert backend.get_status("ray-sandbox-x") == SandboxStatus.TERMINATED
    assert (
        backend.get_termination_reason("ray-sandbox-x")
        == "Sandbox exceeded its enforced memory limit"
    )
    with pytest.raises(SandboxError, match="exceeded its enforced memory limit"):
        backend.get_config("ray-sandbox-x")


@pytest.mark.parametrize("same_job", [False, True])
def test_reclaim_cancels_only_current_jobs_step(
    monkeypatch, slurm_job, tmp_path, same_job
):
    backend = GVisorSandboxBackend(image_manager=Mock())
    config = GVisorSandboxConfig(image="busybox:latest")
    owner = backend._own_bundle(config, str(tmp_path))
    owner.close()
    if not same_job:
        monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.delenv("RAY_SANDBOX_CGROUP_PATH", raising=False)
    monkeypatch.setattr(gvisor.subprocess, "run", Mock())
    cancel = Mock()
    monkeypatch.setattr(slurm_steps, "cancel", cancel)
    assert backend._reclaim_if_unowned(str(tmp_path), "ray-sandbox-000000000001")
    if same_job:
        cancel.assert_called_once_with("ray-sandbox-000000000001")
    else:
        cancel.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
