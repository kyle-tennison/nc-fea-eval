from pathlib import Path

import gmsh
import numpy as np
from loguru import logger

from nc_fea_eval.config import MeshConfig
from nc_fea_eval.profiler import Profiler
from nc_fea_eval.step import StepFaces


class Mesher:
    """Mesh a STEP file with gmsh and tag the fixed/loaded faces as physical groups."""

    FIXED = "fixed"
    LOAD = "load"
    VOLUME = "solid"

    @classmethod
    def mesh(
        cls,
        step_file: Path,
        msh_file: Path,
        fixed_faces: list[str],
        load_faces: list[str],
        mesh: MeshConfig,
        profiler: Profiler,
    ) -> Path:
        with profiler.stage("read STEP faces"):
            step_vertices, step_all = StepFaces.vertices(step_file, fixed_faces + load_faces)

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")  # keep SI units

            with profiler.stage("import geometry + tag faces"):
                logger.info("Importing {}", step_file.name)
                gmsh.model.occ.importShapes(str(step_file))
                gmsh.model.occ.synchronize()

                surfaces = cls._match_surfaces(step_vertices, step_all)
                for name, ids in ((cls.FIXED, fixed_faces), (cls.LOAD, load_faces)):
                    tags = [surfaces[i] for i in ids]
                    logger.info("{} faces: {}", name, ", ".join(f"{i} -> surface {surfaces[i]}" for i in ids))
                    gmsh.model.addPhysicalGroup(2, tags, name=name)
                volumes = [tag for _, tag in gmsh.model.getEntities(3)]
                gmsh.model.addPhysicalGroup(3, volumes, name=cls.VOLUME)

            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh.size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh.size_max)
            with profiler.stage("generate mesh"):
                gmsh.model.mesh.generate(3)
            logger.info("Meshed {} nodes", len(gmsh.model.mesh.getNodes()[0]))

            with profiler.stage("write mesh"):
                gmsh.write(str(msh_file))
            return msh_file
        finally:
            gmsh.finalize()

    @staticmethod
    def _match_surfaces(step_vertices: dict[str, np.ndarray], step_all: np.ndarray) -> dict[str, int]:
        """Map STEP face names to gmsh surface tags by comparing their vertex sets.

        gmsh doesn't import STEP face names, but a face's vertices identify it.
        """
        gmsh_all = np.array([gmsh.model.getValue(0, tag, []) for _, tag in gmsh.model.getEntities(0)])
        span = np.ptp(gmsh_all, axis=0).max()
        scale = span / np.ptp(step_all, axis=0).max()  # STEP file units -> meters
        tol = 1e-6 * span

        surface_vertices = {
            tag: np.array(
                [gmsh.model.getValue(0, p, []) for _, p in gmsh.model.getBoundary([(2, tag)], combined=False, recursive=True)]
            )
            for _, tag in gmsh.model.getEntities(2)
        }

        def same(a: np.ndarray, b: np.ndarray) -> bool:
            d = np.linalg.norm(a[:, None] - b[None], axis=-1)
            return len(np.unique(b, axis=0)) == len(a) and bool((d.min(axis=1) < tol).all())

        result = {}
        for name, verts in step_vertices.items():
            verts = scale * verts
            matches = [tag for tag, sv in surface_vertices.items() if same(verts, sv)]
            if len(matches) != 1:
                raise RuntimeError(f"STEP face {name!r} matched gmsh surfaces {matches}; expected exactly one")
            result[name] = matches[0]
        return result
