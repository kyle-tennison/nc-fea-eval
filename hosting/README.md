# hosting

The campus machine can't accept inbound connections, so it never does. Clients submit jobs to a small
public **proxy** (Lightsail). The campus **dispatcher** polls the proxy over outbound HTTPS, runs each job
in the `fea-eval` Docker image, and uploads the results.

```
client ──HTTPS──▶ proxy (Lightsail, SQLite + disk) ◀──HTTPS poll── dispatch (campus) ──▶ docker fea-eval
```

- **Polling:** every 1 s for an hour after the last job, every 20 s otherwise (`--hot-interval`,
  `--idle-interval`, `--hot-window`).
- **Leases:** a claimed job is leased to the worker, which heartbeats every 15 s. If heartbeats stop for
  `FEA_LEASE_SECONDS` (120 s) the job is requeued; after `FEA_MAX_ATTEMPTS` (3) leases it's marked failed.
  A worker whose lease lapsed gets `409` and drops the job, so it can't clobber the newer run.
- **Two tokens:** `FEA_CLIENT_TOKEN` for submitting/reading jobs, `FEA_WORKER_TOKEN` for the dispatcher.
- **Isolation:** jobs run with `--network none` as the dispatcher's own uid.

## API

| | Auth | |
|---|---|---|
| `POST /jobs` (multipart `input`, `step`) | client | Queue a run. `input`'s `step_file` must be a bare name like `part.step`; the uploaded STEP is stored under it. |
| `GET /jobs/{id}` | client | Status: `queued` → `running` → `done` / `failed` |
| `GET /jobs/{id}/result` | client | Zip of the `results/` dir (`.pvd`, `.vtu`, `metrics.json`, ...) plus `log.txt` |
| `GET /jobs/{id}/log` | client | Pipeline output of a failed job |
| `POST /worker/claim` | worker | Lease the next job (`204` if none) |
| `POST /worker/jobs/{id}/{heartbeat,complete,fail}` | worker | Lease token in `X-Lease` |

```bash
curl -H "Authorization: Bearer $FEA_CLIENT_TOKEN" -F input=@input.toml -F step=@part.step https://52-37-243-198.sslip.io/jobs
curl -H "Authorization: Bearer $FEA_CLIENT_TOKEN" https://52-37-243-198.sslip.io/jobs/<id>
curl -H "Authorization: Bearer $FEA_CLIENT_TOKEN" -o result.zip https://52-37-243-198.sslip.io/jobs/<id>/result
```

## Deploying

**Proxy (Lightsail):** deployed at `https://52-37-243-198.sslip.io`, behind the box's existing nginx.

1. Clone the repo to `/opt/fea-eval`, install `uv`, then run `uv sync --frozen --no-dev` in `hosting/proxy`.
2. Write `/etc/fea-proxy.env` with `FEA_CLIENT_TOKEN`, `FEA_WORKER_TOKEN` (generate them with
   `openssl rand -hex 32`) and `FEA_DATA_DIR=/var/lib/fea-proxy`.
3. Install [`deploy/fea-proxy.service`](proxy/deploy/fea-proxy.service) and
   [`deploy/nginx.conf`](proxy/deploy/nginx.conf), then run `certbot --nginx -d <name> --redirect` for TLS.
   Uvicorn only listens on localhost.

**Dispatcher (campus):** from the repo checkout on the campus machine, run

```bash
hosting/dispatch/deploy/install.sh
```

It builds the `fea-eval` image, installs the dispatcher, asks for the worker token once (saved to
`/etc/fea-dispatch.env`), checks it against the proxy, and installs a `fea-dispatch` systemd service that
starts on boot. Without sudo it falls back to a user service with lingering. After a `git pull`, re-run it
to update.

Job files live under `FEA_DATA_DIR/jobs/<id>/` forever; prune old ones with a cron job if disk gets tight.
