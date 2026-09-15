from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from loguru import logger
from mpi4py import MPI

from fea_eval.mesher import Mesher


class Solver:
    """Linear elasticity: clamp one face, apply a uniform traction to another."""

    @classmethod
    def solve(
        cls,
        msh_file: Path,
        out_file: Path,
        traction: tuple[float, float, float],
        youngs_modulus: float,
        poisson_ratio: float,
        degree: int = 2,
    ) -> Path:
        data = io.gmsh.read_from_msh(msh_file, MPI.COMM_WORLD, rank=0, gdim=3)
        domain, facet_tags = data.mesh, data.facet_tags
        fixed_tag = data.physical_groups[Mesher.FIXED].tag
        load_tag = data.physical_groups[Mesher.LOAD].tag
        logger.info("Loaded mesh with {} cells", domain.topology.index_map(3).size_global)

        V = fem.functionspace(domain, ("Lagrange", degree, (3,)))

        fdim = domain.topology.dim - 1
        fixed_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(fixed_tag))
        bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)

        mu = youngs_modulus / (2 * (1 + poisson_ratio))
        lmbda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        def sigma(w):
            eps = ufl.sym(ufl.grad(w))
            return 2 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(3)

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        t = fem.Constant(domain, np.asarray(traction, dtype=np.float64))

        a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
        L = ufl.dot(t, v) * ds(load_tag)

        logger.info("Solving ({} dofs)", V.dofmap.index_map.size_global * 3)
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="fea_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        )
        uh = problem.solve()
        logger.info("Max displacement: {:.4e} m", np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1).max())

        cls._write(uh, sigma, out_file)
        return out_file

    @staticmethod
    def _write(uh: fem.Function, sigma, out_file: Path) -> None:
        domain = uh.function_space.mesh

        # XDMF needs output on the mesh's own (P1) geometry.
        u_out = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))), name="displacement")
        u_out.interpolate(uh)

        s = sigma(uh)
        dev = s - ufl.tr(s) / 3 * ufl.Identity(3)
        W = fem.functionspace(domain, ("DG", 0))
        von_mises = fem.Function(W, name="von_mises")
        von_mises.interpolate(fem.Expression(ufl.sqrt(1.5 * ufl.inner(dev, dev)), W.element.interpolation_points))
        logger.info("Max von Mises: {:.4e} Pa", von_mises.x.array.max())

        with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u_out)
            xdmf.write_function(von_mises)
        logger.success("Wrote {}", out_file)
