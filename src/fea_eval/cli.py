import tempfile
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger


class Cli:
    @staticmethod
    def pipeline(
        step_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input STEP file")],
        output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("results"),
        traction: Annotated[
            tuple[float, float, float], typer.Option(help="Traction on the load face [Pa]")
        ] = (0.0, 0.0, -1e6),
        youngs_modulus: Annotated[float, typer.Option(help="Young's modulus [Pa]")] = 200e9,
        poisson_ratio: Annotated[float, typer.Option(help="Poisson's ratio")] = 0.3,
        mesh_size: Annotated[float | None, typer.Option(help="Max element size [m]")] = None,
    ) -> None:
        """Mesh STEP_FILE, clamp one end, load the other, and write an XDMF result."""
        # Deferred so `--help` is fast and doesn't need dolfinx.
        from fea_eval.mesher import Mesher
        from fea_eval.solver import Solver

        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fea-") as tmp:
            logger.info("Working in {}", tmp)
            msh = Mesher.mesh(step_file, Path(tmp) / "mesh.msh", mesh_size)
            result = Solver.solve(msh, output / f"{step_file.stem}.xdmf", traction, youngs_modulus, poisson_ratio)
        logger.success("Done. Open {} in ParaView", result)

    @classmethod
    def run(cls) -> None:
        app = typer.Typer(add_completion=False)
        app.command()(cls.pipeline)
        app()
