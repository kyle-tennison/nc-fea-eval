import base64
import hmac
import os
import re
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from nc_fea_proxy.store import Job, Status, Store

STEP_NAME = re.compile(r"\w[\w.-]*\.(step|stp)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    client_token: str  # whoever submits jobs
    worker_token: str  # the campus dispatcher
    lease_seconds: float
    max_attempts: int
    max_step_bytes: int
    max_result_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("FEA_DATA_DIR", "data")),
            client_token=os.environ["FEA_CLIENT_TOKEN"],
            worker_token=os.environ["FEA_WORKER_TOKEN"],
            lease_seconds=float(os.environ.get("FEA_LEASE_SECONDS", "120")),
            max_attempts=int(os.environ.get("FEA_MAX_ATTEMPTS", "3")),
            max_step_bytes=int(float(os.environ.get("FEA_MAX_STEP_MB", "5")) * 2**20),
            max_result_bytes=int(float(os.environ.get("FEA_MAX_RESULT_MB", "250")) * 2**20),
        )


class FailReport(BaseModel):
    error: str
    log: str = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir, settings.lease_seconds, settings.max_attempts)
    app = FastAPI(title="fea-eval proxy")
    client = _bearer(settings.client_token)
    worker = _bearer(settings.worker_token)

    def get_job(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
        return job

    def lease_lost() -> HTTPException:
        return HTTPException(status.HTTP_409_CONFLICT, "lease is no longer held")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    # --- clients ---

    @app.post("/jobs", status_code=status.HTTP_201_CREATED, dependencies=[client])
    async def submit(input: UploadFile, step: UploadFile) -> dict:
        """Queue a run: `input` is the input TOML, `step` is the file its `step_file` names."""
        raw_toml = await input.read()
        try:
            config = tomllib.loads(raw_toml.decode())
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"bad input TOML: {e}")
        step_name = config.get("step_file")
        if not isinstance(step_name, str) or not STEP_NAME.fullmatch(step_name):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "step_file must be a bare file name like part.step"
            )
        step_bytes = await step.read(settings.max_step_bytes + 1)
        if len(step_bytes) > settings.max_step_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "STEP file too large")
        return _public(store.submit(raw_toml, step_name, step_bytes))

    @app.get("/jobs/{job_id}", dependencies=[client])
    def job_status(job_id: str) -> dict:
        store.expire_leases()
        return _public(get_job(job_id))

    @app.get("/jobs/{job_id}/result", dependencies=[client])
    def job_result(job_id: str) -> FileResponse:
        path = store.job_dir(get_job(job_id).id) / "results.zip"
        if not path.exists():
            raise HTTPException(status.HTTP_409_CONFLICT, "job has no results")
        return FileResponse(path, media_type="application/zip", filename=f"{job_id}.zip")

    @app.get("/jobs/{job_id}/log", dependencies=[client])
    def job_log(job_id: str) -> PlainTextResponse:
        """Pipeline output of a failed job (a finished job's log is inside its results zip)."""
        path = store.job_dir(get_job(job_id).id) / "log.txt"
        if not path.exists():
            raise HTTPException(status.HTTP_409_CONFLICT, "job has no log")
        return PlainTextResponse(path.read_text())

    # --- the dispatcher ---

    @app.post("/worker/claim", dependencies=[worker])
    def claim() -> Response:
        lease = store.claim()
        if lease is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        job_dir = store.job_dir(lease.job.id)
        return JSONResponse(
            {
                "id": lease.job.id,
                "lease": lease.token,
                "step_name": lease.job.step_name,
                "input_toml": (job_dir / "input.toml").read_text(),
                "step_b64": base64.b64encode((job_dir / lease.job.step_name).read_bytes()).decode(),
            }
        )

    @app.post("/worker/jobs/{job_id}/heartbeat", status_code=status.HTTP_204_NO_CONTENT, dependencies=[worker])
    def heartbeat(job_id: str, lease: Annotated[str, Header(alias="X-Lease")]) -> None:
        if not store.heartbeat(job_id, lease):
            raise lease_lost()

    @app.post("/worker/jobs/{job_id}/complete", status_code=status.HTTP_204_NO_CONTENT, dependencies=[worker])
    async def complete(job_id: str, request: Request, lease: Annotated[str, Header(alias="X-Lease")]) -> None:
        """Body is the results zip."""
        get_job(job_id)
        if int(request.headers.get("content-length", 0)) > settings.max_result_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "results too large")
        if not store.complete(job_id, lease, await request.body()):
            raise lease_lost()

    @app.post("/worker/jobs/{job_id}/fail", status_code=status.HTTP_204_NO_CONTENT, dependencies=[worker])
    def fail(job_id: str, report: FailReport, lease: Annotated[str, Header(alias="X-Lease")]) -> None:
        get_job(job_id)
        if not store.fail(job_id, lease, report.error, report.log):
            raise lease_lost()

    return app


def _bearer(token: str):
    scheme = HTTPBearer()

    def check(credentials: Annotated[HTTPAuthorizationCredentials, Depends(scheme)]) -> None:
        if not hmac.compare_digest(credentials.credentials.encode(), token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad token")

    return Depends(check)


def _public(job: Job) -> dict:
    def iso(ts: float | None) -> str | None:
        return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None

    return {
        "id": job.id,
        "status": job.status,
        "created_at": iso(job.created_at),
        "started_at": iso(job.claimed_at),
        "finished_at": iso(job.finished_at),
        "attempts": job.attempts,
        "error": job.error,
        "result_ready": job.status == Status.DONE,
    }
