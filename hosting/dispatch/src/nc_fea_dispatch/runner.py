import glob
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger


@dataclass(frozen=True, slots=True)
class Outcome:
    error: str | None  # None on success
    log: Path


class Runner:
    """Runs `fea-pipeline input.toml -o results` in the fea-eval image against a job directory."""

    def __init__(
        self, image: str, timeout: float, heartbeat_every: float, docker_args: list[str], gpus: str = "auto"
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.heartbeat_every = heartbeat_every
        self.docker_args = docker_args
        self.gpu_args = self._gpu_args(gpus)

    def run(self, job_id: str, workdir: Path, heartbeat: Callable[[], None]) -> Outcome:
        """Block until the pipeline exits, calling `heartbeat` periodically. Raises LeaseLost (after
        killing the container) if a heartbeat says the job was reassigned."""
        container = f"fea-job-{job_id}"
        log = workdir / "log.txt"
        command = [
            "docker", "run", "--rm", "--name", container,
            "--network", "none",  # inputs come from the internet; the solver needs no network
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",  # results stay ours; JIT cache writable
            "-v", f"{workdir.resolve()}:/work",
            # Tell the pipeline which solver to take, so a job that can't reach the GPU
            # we promised it fails with that as the reason instead of quietly going slow.
            "-e", f"FEA_SOLVER={'gpu' if self.gpu_args else 'cpu'}",
            *self.gpu_args,
            *self.docker_args,
            self.image, "input.toml", "--output", "results",
        ]  # fmt: skip
        logger.info("Running job {} in {}", job_id, container)
        started = time.monotonic()
        with log.open("wb") as out:
            process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=self.heartbeat_every)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() - started > self.timeout:
                    self._kill(container, process)
                    return Outcome(f"timed out after {self.timeout / 60:.0f} min", log)
                self._beat(job_id, heartbeat)
        except BaseException:  # LeaseLost, Ctrl-C, ...
            self._kill(container, process)
            raise

        logger.info("Job {} exited {} after {:.0f} s", job_id, code, time.monotonic() - started)
        if code != 0:
            return Outcome(f"pipeline exited with code {code}", log)
        if not (workdir / "results").is_dir():
            return Outcome("pipeline wrote no results", log)
        return Outcome(None, log)

    def _gpu_args(self, gpus: str) -> list[str]:
        """The `docker run` arguments that give a job the GPUs, or [] to run on the CPU.

        `--gpus` is the clean way, but it needs the NVIDIA Container Toolkit, which is
        root to install. Where it's missing we hand the container the driver's device
        nodes and `libcuda` directly, which does the same job for a compute-only run and
        needs nothing but membership of the docker group. Either way a throwaway
        container has to prove CUDA works before any job is told to rely on it, since
        otherwise every job would fail at the solve.
        """
        if gpus in ("", "none"):
            return []
        if gpus != "auto":
            return ["--gpus", gpus]

        for name, args in (("--gpus all", ["--gpus", "all"]), ("driver device nodes", self._driver_mounts())):
            if not args:
                continue
            probe = self._probe(args)
            if probe is not None:
                logger.info("Jobs get the GPUs via {}: {}", name, probe)
                return args
            logger.debug("GPUs not reachable via {}", name)
        logger.warning("No GPUs reachable from containers, so jobs will solve on the CPU")
        return []

    @staticmethod
    def _driver_mounts() -> list[str]:
        """Device nodes plus the host's driver libraries, mounted where the loader looks."""
        nodes = sorted(glob.glob("/dev/nvidia[0-9]*")) + [
            node for node in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools") if os.path.exists(node)
        ]
        if not nodes:
            return []
        args = [arg for node in nodes for arg in ("--device", node)]
        # Bind the versioned file onto the soname: the container has no driver package,
        # so nothing else provides the .so.1 that libcudart and nvidia-smi dlopen.
        for soname in ("libcuda.so.1", "libnvidia-ml.so.1"):
            found = glob.glob(f"/usr/lib/x86_64-linux-gnu/{soname.split('.so')[0]}.so.*.*")
            if not found:
                return []
            args += ["-v", f"{max(found)}:/usr/lib/x86_64-linux-gnu/{soname}:ro"]
        return args

    def _probe(self, args: list[str]) -> str | None:
        """What CUDA reports from inside a container run with `args`, or None if it can't."""
        script = "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'device(s)')"
        probe = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", *args, "--entrypoint", "python3", self.image, "-c", script],
            capture_output=True,
            text=True,
        )
        return probe.stdout.strip() if probe.returncode == 0 else None

    @staticmethod
    def _beat(job_id: str, heartbeat: Callable[[], None]) -> None:
        try:
            heartbeat()  # LeaseLost propagates
        except httpx.HTTPError as e:  # a blip; the lease outlives a few missed beats
            logger.warning("Heartbeat for job {} failed: {}", job_id, e)

    @staticmethod
    def _kill(container: str, process: subprocess.Popen) -> None:
        subprocess.run(["docker", "kill", container], check=False, capture_output=True)
        process.wait()
