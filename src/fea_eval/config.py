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
    fixed_faces: list[str]  # STEP face names, e.g. "f_4"
    load_faces: list[str]
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
            fixed_faces=cls._names(raw["fixed"]["faces"], "fixed.faces"),
            load_faces=cls._names(raw["load"]["faces"], "load.faces"),
            force=force,
            mesh=MeshConfig(**raw["mesh"]),
            material=MaterialConfig(**raw["material"]),
        )

    @staticmethod
    def _names(value: object, key: str) -> list[str]:
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ValueError(f'{key} must be a non-empty list of STEP face names, e.g. ["f_4"]; got {value!r}')
        return value
