# fea-eval

STEP → gmsh → dolfinx linear elasticity. dolfinx isn't on PyPI, so it runs in the
`dolfinx/dolfinx` image, with uv installing this project on top.

```bash
docker build -t fea-eval .
docker run --rm -v "$PWD:/work" fea-eval input.toml -o results
```

Open `results/named-faces.pvd` in ParaView.

Everything is configured in [`input.toml`](input.toml) (SI units): the STEP path,
fixed/loaded faces, total load force [N], mesh sizes, and material.

Faces are STEP `ADVANCED_FACE` entity IDs (the `#98` in `#98=ADVANCED_FACE(...)`).
gmsh drops these IDs, so each is matched to a gmsh surface by its vertex coordinates.
