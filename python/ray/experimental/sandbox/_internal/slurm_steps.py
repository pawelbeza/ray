"""Per-sandbox limits from Slurm job steps, for nodes without delegated cgroups.

With ``RAY_SANDBOX_SLURM_STEPS=1`` in a Slurm job, each sandbox's run tree starts
in its own ``srun --overlap`` step on this node, named after the sandbox. The
step's cgroups cap the sandbox's memory, CPUs (cpuset) and GPU devices, and an
out-of-memory kill takes the step, never the job. A worker inside a container
(Pyxis/enroot) has the step re-enter the worker's user and mount namespaces, so
the sandbox sees the same files the worker does; the container then needs the
host's Slurm client (``srun``, ``squeue``, ``scancel``) and munge socket.
"""

import logging
import math
import os
import socket
import subprocess
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

SLURM_STEPS_ENV = "RAY_SANDBOX_SLURM_STEPS"

# The initial user namespace has this fixed inode on every Linux kernel.
_INIT_USER_NS = "user:[4026531837]"
# A worker started by a bound step would hand its binding to the sandbox's
# step, which Slurm then rejects as outside that step's own CPUs.
_INHERITED_BINDING = (
    "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_LIST",
    "SLURM_CPU_BIND_TYPE",
    "SLURM_CPU_BIND_VERBOSE",
    "SLURM_MEM_BIND",
    "SLURM_MEM_BIND_LIST",
    "SLURM_MEM_BIND_TYPE",
)
# A job submitted with `--gpus` exports its GPU request, which srun reads as the step's
# own and refuses to combine with the step's per-node `--gres`.
_INHERITED_GPU_REQUEST = (
    "SLURM_GPUS",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS_PER_SOCKET",
    "SLURM_GPUS_PER_TASK",
    "SLURM_CPUS_PER_GPU",
    "SLURM_MEM_PER_GPU",
)
# What slurmstepd and srun print when the kernel OOM-kills a step's task.
_OOM_MARKERS = ("oom_kill event", "Out Of Memory")


def enabled() -> bool:
    """Whether sandboxes on this worker run in their own Slurm steps."""
    return os.environ.get(SLURM_STEPS_ENV) == "1" and "SLURM_JOB_ID" in os.environ


def step_command(
    argv: Sequence[str],
    *,
    name: str,
    cpu: Optional[float],
    memory: Optional[int],
    gpus: bool,
) -> List[str]:
    """Wrap `argv` to run as this node's Slurm step `name`, capped at `cpu` cores
    and `memory` bytes. A GPU sandbox's step may open every GPU of the job on this
    node; the sandbox's own spec still exposes only its assigned GPUs.
    """
    command = ["env"]
    for variable in (*_INHERITED_BINDING, *_INHERITED_GPU_REQUEST):
        command += ["-u", variable]
    command += [
        "srun",
        "--overlap",
        f"--jobid={os.environ['SLURM_JOB_ID']}",
        f"--nodelist={os.environ.get('SLURMD_NODENAME') or socket.gethostname()}",
        "--nodes=1",
        "--ntasks=1",
        "--quiet",
        "--mpi=none",
        "--cpu-bind=none",
        "--chdir=/",
        f"--job-name={name}",
        f"--cpus-per-task={max(1, math.ceil(cpu or 1))}",
        # Per-node GRES, not `--gpus`, which Slurm refused for steps on a multi-node job's
        # nodes other than the batch host, even with `--overlap`.
        (
            f"--gres=gpu:{os.environ.get('SLURM_GPUS_ON_NODE', '0')}"
            if gpus
            else "--gres=none"
        ),
    ]
    if memory:
        command.append(f"--mem={math.ceil(memory / 2**20)}M")
    command.append("--")
    if os.readlink("/proc/self/ns/user") != _INIT_USER_NS:
        command += [
            "nsenter",
            f"--target={os.getpid()}",
            "--user",
            "--mount",
            "--preserve-credentials",
            "--",
        ]
    return command + list(argv)


def oom_killed(stderr_log_path: str) -> bool:
    """Whether Slurm reported an out-of-memory kill of the sandbox's step."""
    try:
        with open(stderr_log_path, encoding="utf-8", errors="replace") as log:
            text = log.read()
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Failed to read Slurm sandbox stderr log %s", stderr_log_path)
        raise
    return any(marker in text for marker in _OOM_MARKERS)


def cancel(name: str) -> None:
    """Best-effort cancel of this job's step `name`, so nothing of it outlives a delete."""
    try:
        steps = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--steps",
                f"--jobs={os.environ['SLURM_JOB_ID']}",
                "--format=%i %j",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        for line in steps.splitlines():
            step_id, _, step_name = line.strip().partition(" ")
            if step_name == name:
                subprocess.run(
                    ["scancel", "--signal=KILL", step_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
    except (OSError, subprocess.SubprocessError) as err:
        logger.warning("Failed to cancel Slurm sandbox step %s: %s", name, err)
