import io
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePath

import httpx
from loguru import logger

from nc_fea_dispatch.client import Claim, LeaseLost, ProxyClient
from nc_fea_dispatch.runner import Runner

LOG_TAIL_BYTES = 64 * 1024


class Dispatcher:
    """Polls the proxy fast while jobs have been coming in, slowly once things go quiet."""

    def __init__(
        self,
        proxy: ProxyClient,
        runner: Runner,
        work_root: Path,
        hot_interval: float,
        idle_interval: float,
        hot_window: float,
    ) -> None:
        self.proxy = proxy
        self.runner = runner
        self.work_root = work_root
        self.hot_interval = hot_interval
        self.idle_interval = idle_interval
        self.hot_window = hot_window
        self.last_activity: float | None = None  # monotonic; None = idle since startup

    def run_forever(self) -> None:
        logger.info("Polling every {} s when active, {} s when idle", self.hot_interval, self.idle_interval)
        while True:
            try:
                claim = self.proxy.claim()
            except httpx.HTTPError as e:
                logger.warning("Poll failed: {}", e)
                time.sleep(self.idle_interval)
                continue
            if claim is None:
                time.sleep(self.interval())
                continue
            self.last_activity = time.monotonic()
            self.process(claim)
            self.last_activity = time.monotonic()

    def interval(self) -> float:
        hot = self.last_activity is not None and time.monotonic() - self.last_activity < self.hot_window
        return self.hot_interval if hot else self.idle_interval

    def process(self, claim: Claim) -> None:
        if PurePath(claim.step_name).name != claim.step_name:  # the proxy checks this too
            logger.error("Job {} has a suspicious STEP name {!r}", claim.id, claim.step_name)
            return self._report_failure(claim, "invalid step file name", "")

        self.work_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix=f"fea-{claim.id[:8]}-", dir=self.work_root))
        try:
            (workdir / "input.toml").write_text(claim.input_toml)
            (workdir / claim.step_name).write_bytes(claim.step)
            outcome = self.runner.run(claim.id, workdir, lambda: self.proxy.heartbeat(claim))
            if outcome.error is None:
                self.proxy.complete(claim, self._zip_results(workdir))
                logger.success("Job {} done", claim.id)
            else:
                logger.warning("Job {} failed: {}", claim.id, outcome.error)
                self._report_failure(claim, outcome.error, self._tail(outcome.log))
        except LeaseLost:
            logger.warning("Lost the lease on job {}; dropped it", claim.id)
        except httpx.HTTPError as e:
            logger.error("Couldn't report job {} ({}); the proxy will requeue it", claim.id, e)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _report_failure(self, claim: Claim, error: str, log: str) -> None:
        try:
            self.proxy.fail(claim, error, log)
        except (LeaseLost, httpx.HTTPError) as e:
            logger.error("Couldn't report failure of job {}: {!r}", claim.id, e)

    @staticmethod
    def _zip_results(workdir: Path) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((workdir / "results").rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(workdir / "results"))
            archive.write(workdir / "log.txt", "log.txt")
        return buffer.getvalue()

    @staticmethod
    def _tail(log: Path) -> str:
        data = log.read_bytes()
        return data[-LOG_TAIL_BYTES:].decode(errors="replace")
