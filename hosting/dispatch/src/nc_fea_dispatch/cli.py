import tempfile
from pathlib import Path
from typing import Annotated

import typer

from nc_fea_dispatch.client import ProxyClient
from nc_fea_dispatch.dispatcher import Dispatcher
from nc_fea_dispatch.runner import Runner

DEFAULT_WORK_DIR = Path(tempfile.gettempdir(), "fea-dispatch")


class Cli:
    @staticmethod
    def dispatch(
        url: Annotated[str, typer.Option(envvar="FEA_PROXY_URL", help="Proxy base URL, e.g. https://fea.example.com")],
        token: Annotated[str, typer.Option(envvar="FEA_WORKER_TOKEN", help="Worker bearer token")],
        image: Annotated[str, typer.Option(envvar="FEA_IMAGE", help="fea-eval Docker image")] = "fea-eval",
        work_dir: Annotated[Path, typer.Option(help="Where job directories are staged")] = DEFAULT_WORK_DIR,
        hot_interval: Annotated[float, typer.Option(help="Seconds between polls after recent activity")] = 1,
        idle_interval: Annotated[float, typer.Option(help="Seconds between polls once idle")] = 20,
        hot_window: Annotated[float, typer.Option(help="Minutes after a job that polling stays fast")] = 60,
        timeout: Annotated[float, typer.Option(help="Kill a run after this many minutes")] = 60,
        heartbeat: Annotated[
            float, typer.Option(help="Seconds between heartbeats; keep well under the proxy lease")
        ] = 15,
        docker_arg: Annotated[
            list[str] | None, typer.Option(help="Extra `docker run` argument, repeatable (e.g. --docker-arg=--cpus=8)")
        ] = None,
    ) -> None:
        """Poll the proxy for fea-eval jobs and run them locally in Docker."""
        runner = Runner(image, timeout * 60, heartbeat, docker_arg or [])
        Dispatcher(
            ProxyClient(url, token), runner, work_dir, hot_interval, idle_interval, hot_window * 60
        ).run_forever()

    @classmethod
    def run(cls) -> None:
        app = typer.Typer(add_completion=False)
        app.command()(cls.dispatch)
        app()
