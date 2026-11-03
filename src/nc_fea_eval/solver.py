import os
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from loguru import logger
from mpi4py import MPI

from nc_fea_eval.config import LoadConfig, MaterialConfig
from nc_fea_eval.load import LoadFaces
from nc_fea_eval.mesher import Mesher
from nc_fea_eval.profiler import Profiler


class Solver:
    """Linear elasticity: clamp some faces, load others with a force, torque or pressure."""

    @classmethod
    def solve(
        cls,
        msh_file: Path,
        out_file: Path,
        load: LoadConfig,
        material: MaterialConfig,
        profiler: Profiler,
        degree: int = 2,
    ) -> Path:
        with profiler.stage("read mesh"):
            data = io.gmsh.read_from_msh(msh_file, MPI.COMM_WORLD, rank=0, gdim=3)
        domain, facet_tags = data.mesh, data.facet_tags
        fixed_tag = data.physical_groups[Mesher.FIXED].tag
        load_tag = data.physical_groups[Mesher.LOAD].tag
        logger.info("Loaded mesh with {} cells", domain.topology.index_map(3).size_global)

        with profiler.stage("setup + compile forms"):
            V = fem.functionspace(domain, ("Lagrange", degree, (3,)))

            fdim = domain.topology.dim - 1
            fixed_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(fixed_tag))
            bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)

            E, nu = material.youngs_modulus, material.poisson_ratio
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

            def sigma(w):
                eps = ufl.sym(ufl.grad(w))
                return 2 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(3)

            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
            ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
            faces = LoadFaces(domain, ds(load_tag), load)
            t = faces.traction()

            a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
            L = ufl.dot(t, v) * ds(load_tag)

            logger.info("Solving ({} dofs)", V.dofmap.index_map.size_global * 3)
            device = cls._gpu_device()
            if device is None:
                problem = LinearProblem(
                    a,
                    L,
                    bcs=[bc],
                    petsc_options_prefix="fea_",
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                )
        with profiler.stage("assemble + solve"):
            uh = cls._solve_gpu(a, L, bc, V, device) if device is not None else problem.solve()
        logger.info("Max displacement: {:.4e} m", np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1).max())

        with profiler.stage("post-process + write"):
            cls._write(uh, sigma, out_file)
            faces.write_response(uh, out_file.parent / "load_faces.json")
        return out_file

    @staticmethod
    def _gpu_device() -> int | None:
        """The CUDA device to factorize on, or None to stay on CPU MUMPS.

        FEA_SOLVER=gpu demands the GPU (and fails loudly if it isn't usable), cpu forces
        MUMPS, and the default auto uses a GPU when the container actually has one --- so
        the same image runs on a workstation with cards and on a host without them.
        FEA_GPU_DEVICE picks the card when several are visible.
        """
        choice = os.environ.get("FEA_SOLVER", "auto").lower()
        if choice not in ("auto", "cpu", "gpu"):
            raise ValueError(f"FEA_SOLVER must be auto, cpu or gpu, not {choice!r}")
        if choice == "cpu":
            return None

        device = int(os.environ.get("FEA_GPU_DEVICE", "0"))
        try:
            import cupy as cp

            count = cp.cuda.runtime.getDeviceCount()
        except Exception as e:  # no cupy, no driver, no device nodes in the container
            if choice == "gpu":
                raise RuntimeError(f"FEA_SOLVER=gpu but CUDA is unusable: {e}") from e
            logger.info("No usable GPU ({}); solving with MUMPS on the CPU", e)
            return None
        if device >= count:
            raise ValueError(f"FEA_GPU_DEVICE={device} but only {count} CUDA device(s) are visible")

        name = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
        free, total = cp.cuda.Device(device).mem_info
        logger.info("Solving on GPU {} ({}, {:.0f} of {:.0f} GB free)", device, name, free / 2**30, total / 2**30)
        return device

    @staticmethod
    def _mtlayer() -> str | None:
        """cuDSS's OpenMP threading layer, which it loads by path or not at all.

        Reordering and symbolic analysis stay on the CPU, and cuDSS warns that they run
        single-threaded without this; the library ships beside the wheel's other cu12 .so
        files, so look there rather than expecting it on the loader path.
        """
        import glob
        import site

        for root in [*site.getsitepackages(), site.getusersitepackages()]:
            for lib in sorted(glob.glob(f"{root}/nvidia/cu*/lib/libcudss_mtlayer_gomp.so*")):
                return lib
        logger.debug("No cuDSS OpenMP layer found; its CPU-side analysis will be single-threaded")
        return None

    @classmethod
    def _solve_gpu(cls, a, L, bc, V, device: int) -> fem.Function:
        """Direct sparse factorization on the GPU via cuDSS.

        dolfinx still assembles on the CPU; only the factorization and the triangular
        solves move to the device. Keeping a *direct* solve rather than switching to an
        iterative one means this is the same computation MUMPS does, so the displacement
        and stress fields stay comparable to a CPU run of the same job.

        FEA_GPU_HYBRID=1 lets cuDSS keep the factor in host memory, which is slower but
        is the only way through a job whose factor outgrows VRAM.
        """
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
        from nvmath.sparse.advanced import (
            DirectSolver,
            DirectSolverMatrixType,
            DirectSolverMatrixViewType,
            DirectSolverOptions,
            ExecutionCUDA,
            HybridMemoryModeOptions,
        )
        from petsc4py import PETSc

        cp.cuda.Device(device).use()

        a_form, L_form = fem.form(a), fem.form(L)
        A = assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        b = assemble_vector(L_form)
        apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        indptr, indices, data = A.getValuesCSR()
        csr_gb = (data.nbytes + indices.nbytes + indptr.nbytes) / 2**30
        logger.info("Handing cuDSS {} rows, {} nonzeros ({:.2f} GB CSR)", A.getSize()[0], len(data), csr_gb)

        A_gpu = cusparse.csr_matrix(
            (cp.asarray(data), cp.asarray(indices), cp.asarray(indptr)), shape=A.getSize()
        )
        b_gpu = cp.asarray(b.getArray(readonly=True).copy())
        A.destroy()
        b.destroy()

        hybrid = os.environ.get("FEA_GPU_HYBRID") == "1"
        # GENERAL + FULL is an unsymmetric LU over the whole matrix, which is what PETSc
        # asks MUMPS for on the CPU path.
        options = DirectSolverOptions(
            sparse_system_type=DirectSolverMatrixType.GENERAL,
            sparse_system_view=DirectSolverMatrixViewType.FULL,
            multithreading_lib=cls._mtlayer(),
        )
        execution = ExecutionCUDA(
            device_id=device,
            hybrid_memory_mode_options=HybridMemoryModeOptions(hybrid_memory_mode=hybrid),
        )
        with DirectSolver(A_gpu, b_gpu, options=options, execution=execution) as solver:
            solver.plan()
            solver.factorize()
            x_gpu = solver.solve()

        uh = fem.Function(V)
        uh.x.array[:] = cp.asnumpy(x_gpu)[: uh.x.array.size]
        uh.x.scatter_forward()

        del A_gpu, b_gpu, x_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return uh

    @staticmethod
    def _write(uh: fem.Function, sigma, out_file: Path) -> None:
        domain = uh.function_space.mesh

        # Output on the mesh's own (P1) geometry.
        u_out = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))), name="displacement")
        u_out.interpolate(uh)

        s = sigma(uh)
        dev = s - ufl.tr(s) / 3 * ufl.Identity(3)
        W = fem.functionspace(domain, ("DG", 0))
        von_mises = fem.Function(W, name="von_mises")
        von_mises.interpolate(fem.Expression(ufl.sqrt(1.5 * ufl.inner(dev, dev)), W.element.interpolation_points))
        logger.info("Max von Mises: {:.4e} Pa", von_mises.x.array.max())

        # One grid carrying both fields; XDMFFile would write a separate mesh copy per function.
        with io.VTKFile(domain.comm, out_file, "w") as vtk:
            vtk.write_function([u_out, von_mises])
        logger.success("Wrote {}", out_file)
