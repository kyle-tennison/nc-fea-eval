import tomllib
from dataclasses import dataclass
from pathlib import Path

type Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MeshConfig:
    size_min: float
    size_max: float


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    youngs_modulus: float
    poisson_ratio: float


@dataclass(frozen=True, slots=True)
class Config:
    """Contents of an `input.toml`. SI units throughout (m, N, Pa)."""

    step_file: Path
    fixed_faces: list[int]
    load_faces: list[int]
    force: Vec3
    mesh: MeshConfig
    material: MaterialConfig

    @classmethod
    def from_toml(cls, path: Path) -> "Config":
        with path.open("rb") as f:
            raw = tomllib.load(f)

        force = tuple(float(c) for c in raw["load"]["force"])
        if len(force) != 3:
            raise ValueError(f"load.force must have 3 components, got {raw['load']['force']}")

        return cls(
            step_file=(path.parent / raw["step_file"]).resolve(),  # relative to the toml
            fixed_faces=list(raw["fixed"]["faces"]),
            load_faces=list(raw["load"]["faces"]),
            force=force,
            mesh=MeshConfig(**raw["mesh"]),
            material=MaterialConfig(**raw["material"]),
        )
