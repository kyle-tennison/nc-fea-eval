from pathlib import Path

import gmsh
from loguru import logger


class Mesher:
    """Mesh a STEP file with gmsh and tag the fixed/loaded faces as physical groups."""

    FIXED = "fixed"
    LOAD = "load"
    VOLUME = "solid"

    @classmethod
    def mesh(cls, step_file: Path, msh_file: Path, mesh_size: float | None = None) -> Path:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")  # keep SI units

            logger.info("Importing {}", step_file.name)
            gmsh.model.occ.importShapes(str(step_file))
            gmsh.model.occ.synchronize()

            fixed, load = cls._select_end_faces()
            gmsh.model.addPhysicalGroup(2, [fixed], name=cls.FIXED)
            gmsh.model.addPhysicalGroup(2, [load], name=cls.LOAD)
            volumes = [tag for _, tag in gmsh.model.getEntities(3)]
            gmsh.model.addPhysicalGroup(3, volumes, name=cls.VOLUME)

            if mesh_size is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
            gmsh.model.mesh.generate(3)
            logger.info("Meshed {} nodes", len(gmsh.model.mesh.getNodes()[0]))

            gmsh.write(str(msh_file))
            return msh_file
        finally:
            gmsh.finalize()

    @staticmethod
    def _select_end_faces() -> tuple[int, int]:
        """Pick the faces at the min/max ends of the part's longest axis.

        Placeholder until the STEP carries named faces.
        """
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        lo, hi = (xmin, ymin, zmin), (xmax, ymax, zmax)
        axis = max(range(3), key=lambda i: hi[i] - lo[i])
        tol = 1e-3 * (hi[axis] - lo[axis])  # OCC boxes are padded

        def at(value: float) -> int:
            matches = [
                tag
                for _, tag in gmsh.model.getEntities(2)
                if abs((box := gmsh.model.getBoundingBox(2, tag))[axis] - value) < tol
                and abs(box[axis + 3] - value) < tol
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one face at {'xyz'[axis]}={value}, got {matches}")
            return matches[0]

        fixed, load = at(lo[axis]), at(hi[axis])
        logger.info("Fixed face {} at {}min, load face {} at {}max", fixed, "xyz"[axis], load, "xyz"[axis])
        return fixed, load
