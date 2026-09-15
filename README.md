# fea-eval

STEP → gmsh → dolfinx linear elasticity. dolfinx isn't on PyPI, so it runs in the
`dolfinx/dolfinx` image, with uv installing this project on top.

```bash
docker build -t fea-eval .
docker run --rm -v "$PWD:/work" fea-eval input.toml -o results
```

Open `results/<step name>.pvd` in ParaView. Per-stage wall time, CPU time and peak
memory are logged at the end of the run and saved to `results/metrics.json`.

Everything is configured in [`input.toml`](input.toml) (SI units): the STEP path,
fixed/loaded faces, total load force [N], mesh sizes, and material.

Faces are referenced by their STEP name (the `f_4` in `#555 = ADVANCED_FACE('f_4',...)`);
see [`input-template.toml`](input-template.toml). gmsh doesn't import face names, so each
named face is matched to a gmsh surface by its vertex coordinates.
