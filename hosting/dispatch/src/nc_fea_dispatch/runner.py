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

    def __init__(self, image: str, timeout: float, heartbeat_every: float, docker_args: list[str]) -> None:
        self.image = image
        self.timeout = timeout
        self.heartbeat_every = heartbeat_every
        self.docker_args = docker_args

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
