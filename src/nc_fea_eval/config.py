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
class LoadConfig:
    """Exactly one of: a total force spread uniformly, a torque about the faces' centroid, or a pressure."""

    force: Vec3 | None = None  # [N]
    torque: Vec3 | None = None  # [N m]
    pressure: float | None = None  # [Pa], normal to the faces, pushing into the part

    @classmethod
    def from_toml(cls, raw: dict) -> "LoadConfig":
        vectors = {key: cls._vec3(raw[key], f"load.{key}") for key in ("force", "torque") if key in raw}
        pressure = float(raw["pressure"]) if "pressure" in raw else None
        if len(vectors) + (pressure is not None) != 1:
            raise ValueError("[load] needs exactly one of `force`, `torque` or `pressure`")
        return cls(**vectors, pressure=pressure)

    @staticmethod
    def _vec3(value: object, key: str) -> Vec3:
        vector = tuple(float(c) for c in value)
        if len(vector) != 3:
            raise ValueError(f"{key} must have 3 components, got {value}")
        return vector


@dataclass(frozen=True, slots=True)
class Config:
    """Contents of an `input.toml`. SI units throughout (m, N, Pa)."""

    step_file: Path
    fixed_faces: list[str]  # STEP face names, e.g. "f_4"
    load_faces: list[str]
    load: LoadConfig
    mesh: MeshConfig
    material: MaterialConfig

    @classmethod
    def from_toml(cls, path: Path) -> "Config":
        with path.open("rb") as f:
            raw = tomllib.load(f)

        return cls(
            step_file=(path.parent / raw["step_file"]).resolve(),  # relative to the toml
            fixed_faces=cls._names(raw["fixed"]["faces"], "fixed.faces"),
            load_faces=cls._names(raw["load"]["faces"], "load.faces"),
            load=LoadConfig.from_toml(raw["load"]),
            mesh=MeshConfig(**raw["mesh"]),
            material=MaterialConfig(**raw["material"]),
        )

    @staticmethod
    def _names(value: object, key: str) -> list[str]:
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ValueError(f'{key} must be a non-empty list of STEP face names, e.g. ["f_4"]; got {value!r}')
        return value
