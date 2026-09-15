import tempfile
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger


class Cli:
    @staticmethod
    def pipeline(
        input_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input TOML file")],
        output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("results"),
    ) -> None:
        """Mesh the STEP file from INPUT_FILE, fix/load its faces, and write a ParaView result."""
        # Deferred so `--help` is fast and doesn't need dolfinx.
        from fea_eval.config import Config
        from fea_eval.mesher import Mesher
        from fea_eval.solver import Solver

        config = Config.from_toml(input_file)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fea-") as tmp:
            logger.info("Working in {}", tmp)
            msh = Mesher.mesh(config.step_file, Path(tmp) / "mesh.msh", config.fixed_faces, config.load_faces, config.mesh)
            result = Solver.solve(msh, output / f"{config.step_file.stem}.pvd", config.force, config.material)
        logger.success("Done. Open {} in ParaView", result)

    @classmethod
    def run(cls) -> None:
        app = typer.Typer(add_completion=False)
        app.command()(cls.pipeline)
        app()
