import base64
import time
from dataclasses import dataclass

import httpx
from loguru import logger


class LeaseLost(Exception):
    """The proxy handed this job to someone else (our heartbeats stopped reaching it)."""


@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    lease: str
    step_name: str
    input_toml: str
    step: bytes


class ProxyClient:
    def __init__(self, base_url: str, token: str, timeout: float = 30) -> None:
        self._http = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)

    def claim(self) -> Claim | None:
        response = self._http.post("/worker/claim")
        if response.status_code == httpx.codes.NO_CONTENT:
            return None
        response.raise_for_status()
        body = response.json()
        return Claim(
            id=body["id"],
            lease=body["lease"],
            step_name=body["step_name"],
            input_toml=body["input_toml"],
            step=base64.b64decode(body["step_b64"]),
        )

    def heartbeat(self, claim: Claim) -> None:
        self._check(self._http.post(f"/worker/jobs/{claim.id}/heartbeat", headers={"X-Lease": claim.lease}))

    def complete(self, claim: Claim, results_zip: bytes) -> None:
        self._report(
            f"/worker/jobs/{claim.id}/complete",
            claim,
            content=results_zip,
            headers={"Content-Type": "application/zip"},
            timeout=300,
        )

    def fail(self, claim: Claim, error: str, log: str) -> None:
        self._report(f"/worker/jobs/{claim.id}/fail", claim, json={"error": error, "log": log})

    def _report(self, url: str, claim: Claim, attempts: int = 4, **kwargs) -> None:
        """POST a job's outcome, retrying through brief outages so a long solve isn't thrown away."""
        headers = {"X-Lease": claim.lease, **kwargs.pop("headers", {})}
        for attempt in range(1, attempts + 1):
            try:
                response = self._http.post(url, headers=headers, **kwargs)
                if not response.is_server_error or attempt == attempts:
                    return self._check(response)
                problem = f"HTTP {response.status_code}"
            except httpx.TransportError as e:
                if attempt == attempts:
                    raise
                problem = str(e)
            logger.warning("Reporting job {} failed ({}); retry {}/{}", claim.id, problem, attempt, attempts - 1)
            time.sleep(5 * attempt)

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.status_code == httpx.codes.CONFLICT:
            raise LeaseLost
        response.raise_for_status()
