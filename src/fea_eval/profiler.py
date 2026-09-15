import json
import os
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    wall_s: float
    cpu_s: float  # user + system time over all threads and child processes (e.g. JIT compiler)
    peak_rss_mb: float  # process high-water mark at the end of the stage

    @property
    def cpu_percent(self) -> float:
        """100% = one core fully busy."""
        return 100 * self.cpu_s / self.wall_s if self.wall_s else 0.0


@dataclass(slots=True)
class Profiler:
    """Record wall time, CPU time and peak memory for named pipeline stages."""

    stages: list[Stage] = field(default_factory=list)
    _wall0: float = field(default_factory=time.perf_counter)
    _cpu0: float = field(default_factory=lambda: Profiler._cpu_time())

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        wall, cpu = time.perf_counter(), self._cpu_time()
        yield
        stage = Stage(name, time.perf_counter() - wall, self._cpu_time() - cpu, self._peak_rss_mb())
        self.stages.append(stage)
        logger.debug("{}: {:.2f} s ({:.0f}% cpu)", name, stage.wall_s, stage.cpu_percent)

    def total(self) -> Stage:
        return Stage("total", time.perf_counter() - self._wall0, self._cpu_time() - self._cpu0, self._peak_rss_mb())

    def report(self, json_file: Path) -> None:
        total = self.total()
        width = max(len(s.name) for s in [*self.stages, total])
        rows = [f"{'stage':<{width}}  {'wall [s]':>9}  {'cpu [s]':>8}  {'cpu %':>6}  {'peak RSS [MB]':>13}"]
        for s in [*self.stages, total]:
            rows.append(f"{s.name:<{width}}  {s.wall_s:>9.2f}  {s.cpu_s:>8.2f}  {s.cpu_percent:>6.0f}  {s.peak_rss_mb:>13.0f}")
        rows.insert(-1, "-" * len(rows[0]))
        logger.info("Performance ({} CPUs available):\n{}", os.cpu_count(), "\n".join(rows))

        json_file.write_text(
            json.dumps(
                {
                    "cpu_count": os.cpu_count(),
                    "stages": [asdict(s) | {"cpu_percent": s.cpu_percent} for s in self.stages],
                    "total": asdict(total) | {"cpu_percent": total.cpu_percent},
                },
                indent=2,
            )
        )
        logger.info("Wrote {}", json_file)

    @staticmethod
    def _cpu_time() -> float:
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    @staticmethod
    def _peak_rss_mb() -> float:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024**2 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux
