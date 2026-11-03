import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    step_name TEXT NOT NULL,
    created_at REAL NOT NULL,
    claimed_at REAL,
    heartbeat_at REAL,
    finished_at REAL,
    lease TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    status: Status
    step_name: str
    created_at: float
    claimed_at: float | None
    finished_at: float | None
    attempts: int
    error: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"],
            status=Status(row["status"]),
            step_name=row["step_name"],
            created_at=row["created_at"],
            claimed_at=row["claimed_at"],
            finished_at=row["finished_at"],
            attempts=row["attempts"],
            error=row["error"],
        )


@dataclass(frozen=True, slots=True)
class Lease:
    """A claimed job. `token` must accompany every heartbeat/complete/fail so a worker whose lease
    expired (and whose job was handed to someone else) can't overwrite the newer run."""

    job: Job
    token: str


class Store:
    """Job state in SQLite, job files under `root/jobs/<id>/`."""

    def __init__(self, root: Path, lease_seconds: float, max_attempts: int) -> None:
        self.root = root
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def submit(self, input_toml: bytes, step_name: str, step: bytes) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self.job_dir(job_id)
        job_dir.mkdir()
        (job_dir / "input.toml").write_bytes(input_toml)
        (job_dir / step_name).write_bytes(step)
        with self._db() as db:
            db.execute(
                "INSERT INTO jobs (id, status, step_name, created_at) VALUES (?, ?, ?, ?)",
                (job_id, Status.QUEUED, step_name, time.time()),
            )
        logger.info("Queued job {} ({})", job_id, step_name)
        return self.get(job_id)

    def get(self, job_id: str) -> Job | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def claim(self) -> Lease | None:
        """Lease the oldest queued job, or None if the queue is empty."""
        now = time.time()
        token = uuid.uuid4().hex
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._expire_leases(db, now)
            row = db.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1", (Status.QUEUED,)
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status = ?, lease = ?, claimed_at = ?, heartbeat_at = ?, attempts = attempts + 1"
                " WHERE id = ?",
                (Status.RUNNING, token, now, now, row["id"]),
            )
        logger.info("Leased job {}", row["id"])
        return Lease(self.get(row["id"]), token)

    def heartbeat(self, job_id: str, token: str) -> bool:
        with self._db() as db:
            cursor = db.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND lease = ? AND status = ?",
                (time.time(), job_id, token, Status.RUNNING),
            )
        return cursor.rowcount == 1

    def complete(self, job_id: str, token: str, results_zip: bytes) -> bool:
        tmp = self.job_dir(job_id) / f"results.{token}.part"
        tmp.write_bytes(results_zip)
        if not self._finish(job_id, token, Status.DONE, error=None):
            tmp.unlink()
            return False
        tmp.rename(self.job_dir(job_id) / "results.zip")
        return True

    def fail(self, job_id: str, token: str, error: str, log: str) -> bool:
        if not self._finish(job_id, token, Status.FAILED, error=error):
            return False
        (self.job_dir(job_id) / "log.txt").write_text(log)
        return True

    def expire_leases(self) -> None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._expire_leases(db, time.time())

    def _finish(self, job_id: str, token: str, status: Status, error: str | None) -> bool:
        with self._db() as db:
            cursor = db.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?, lease = NULL"
                " WHERE id = ? AND lease = ? AND status = ?",
                (status, error, time.time(), job_id, token, Status.RUNNING),
            )
        if cursor.rowcount == 1:
            logger.info("Job {} {}", job_id, status)
        return cursor.rowcount == 1

    def _expire_leases(self, db: sqlite3.Connection, now: float) -> None:
        """Requeue running jobs whose worker went quiet; give up after `max_attempts` leases."""
        stale = now - self.lease_seconds
        given_up = db.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ?, lease = NULL"
            " WHERE status = ? AND heartbeat_at < ? AND attempts >= ?",
            (Status.FAILED, "worker stopped responding", now, Status.RUNNING, stale, self.max_attempts),
        )
        requeued = db.execute(
            "UPDATE jobs SET status = ?, lease = NULL WHERE status = ? AND heartbeat_at < ?",
            (Status.QUEUED, Status.RUNNING, stale),
        )
        if given_up.rowcount or requeued.rowcount:
            logger.warning("Expired leases: {} requeued, {} failed", requeued.rowcount, given_up.rowcount)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        # Autocommit unless a caller opens `BEGIN IMMEDIATE`; a connection per call keeps threads apart.
        db = sqlite3.connect(self.root / "jobs.db", isolation_level=None, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            if db.in_transaction:
                db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()
