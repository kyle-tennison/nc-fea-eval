import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from loguru import logger

from nc_fea_eval.config import LoadConfig


class LoadFaces:
    """The loaded faces: builds their traction and reports how they moved."""

    def __init__(self, domain, measure: ufl.Measure, load: LoadConfig) -> None:
        self.domain, self.measure, self.load = domain, measure, load
        self.x = ufl.SpatialCoordinate(domain)
        self.area = self._integral(fem.Constant(domain, 1.0))
        self.centroid = np.array([self._integral(self.x[i]) for i in range(3)]) / self.area
        if load.torque is not None:
            torque = np.asarray(load.torque, dtype=np.float64)
            self.axis = fem.Constant(domain, torque / np.linalg.norm(torque))
            offset = self.x - fem.Constant(domain, self.centroid)
            self.radial = offset - ufl.dot(offset, self.axis) * self.axis  # perpendicular to the torque axis
            self.polar = self._integral(ufl.dot(self.radial, self.radial))  # polar moment of the face area

    def traction(self):
        """Surface traction [Pa] whose resultant is the configured force, torque or pressure."""
        load = self.load
        logger.info("Load area {:.4e} m^2, centroid {} m", self.area, np.array2string(self.centroid, precision=4))
        if load.force is not None:
            return fem.Constant(self.domain, np.asarray(load.force, dtype=np.float64) / self.area)
        if load.pressure is not None:
            return -load.pressure * ufl.FacetNormal(self.domain)
        # Shear growing linearly with distance from the axis through the centroid: zero net force, moment = torque.
        return float(np.linalg.norm(load.torque)) / self.polar * ufl.cross(self.axis, self.radial)

    def write_response(self, uh: fem.Function, path: Path) -> None:
        """Write the faces' area, centroid, mean displacement and (for torque) least-squares twist angle."""
        response = {
            "area": self.area,  # [m^2]
            "centroid": self.centroid.tolist(),  # [m]
            "mean_displacement": [self._integral(uh[i]) / self.area for i in range(3)],  # [m]
        }
        if self.load.torque is not None:
            response["twist"] = self._integral(ufl.dot(self.axis, ufl.cross(self.radial, uh))) / self.polar  # [rad]
            logger.info("Load-face twist: {:.4e} rad", response["twist"])
        path.write_text(json.dumps(response, indent=2))

    def _integral(self, expr) -> float:
        """Integral of `expr` over the loaded faces, summed across MPI ranks."""
        return self.domain.comm.allreduce(fem.assemble_scalar(fem.form(expr * self.measure)))
