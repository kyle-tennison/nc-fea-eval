# fea-eval

STEP → gmsh → dolfinx linear elasticity. dolfinx isn't on PyPI, so it runs in the
`dolfinx/dolfinx` image, with uv installing this project on top.

```bash
docker build -t fea-eval .
docker run --rm -v "$PWD:/work" fea-eval named-faces.step -o results
```

Open `results/named-faces.xdmf` in ParaView (pick the "Xdmf3 Reader S").

Options (`--help`): `--traction X Y Z` [Pa], `--youngs-modulus`, `--poisson-ratio`, `--mesh-size` [m].

Until the STEP has named faces, the faces at the two ends of the longest axis are
fixed and loaded (`Mesher._select_end_faces`).
