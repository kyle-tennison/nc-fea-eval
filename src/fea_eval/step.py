import re
from pathlib import Path

import numpy as np


class StepFaces:
    """Minimal STEP reader: resolves face entity IDs (e.g. #98) to their B-rep vertex coordinates."""

    FACE_TYPES = frozenset({"ADVANCED_FACE", "FACE_SURFACE"})
    # Topology walked from a face down to its vertices; geometry (PLANE, LINE, ...) is skipped.
    TOPOLOGY_TYPES = frozenset({"FACE_OUTER_BOUND", "FACE_BOUND", "EDGE_LOOP", "ORIENTED_EDGE", "EDGE_CURVE", "VERTEX_LOOP"})

    ENTITY = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*$", re.DOTALL)
    REF = re.compile(r"#(\d+)")

    @classmethod
    def vertices(cls, step_file: Path, face_ids: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Map each face ID to an (n, 3) array of its vertex coordinates, plus all vertices of the part.

        Coordinates are in STEP file units.
        """
        entities = cls._parse(step_file)

        faces = sorted(i for i, (kind, _) in entities.items() if kind in cls.FACE_TYPES)
        result = {}
        for face_id in face_ids:
            if face_id not in faces:
                kind = entities.get(face_id, ("missing",))[0]
                raise ValueError(f"STEP #{face_id} is {kind}, not a face. Face IDs: {faces}")
            result[face_id] = cls._face_vertices(entities, face_id)
        everything = np.array([cls._point(entities, i) for i, (kind, _) in entities.items() if kind == "VERTEX_POINT"])
        return result, everything

    @classmethod
    def _parse(cls, step_file: Path) -> dict[int, tuple[str, str]]:
        text = step_file.read_text()
        data = text[text.index("DATA;") + len("DATA;") : text.rindex("ENDSEC;")]
        data = re.sub(r"/\*.*?\*/", "", data, flags=re.DOTALL)
        # Complex entities `#n=(A() B() ...)` don't match and are ignored; faces never use them.
        return {
            int(m[1]): (m[2], m[3])
            for record in data.split(";")
            if (m := cls.ENTITY.match(record.strip()))
        }

    @classmethod
    def _face_vertices(cls, entities: dict[int, tuple[str, str]], face_id: int) -> np.ndarray:
        stack, seen, points = [face_id], set(), {}
        while stack:
            eid = stack.pop()
            if eid in seen:
                continue
            seen.add(eid)
            kind, args = entities[eid]
            if kind == "VERTEX_POINT":
                points[eid] = cls._point(entities, eid)
                continue
            for ref in map(int, cls.REF.findall(args)):
                if entities.get(ref, ("",))[0] in cls.TOPOLOGY_TYPES | {"VERTEX_POINT"}:
                    stack.append(ref)
        if not points:
            raise ValueError(f"STEP face #{face_id} has no vertices (closed surface); can't match it")
        return np.array(list(points.values()))

    @classmethod
    def _point(cls, entities: dict[int, tuple[str, str]], vertex_id: int) -> tuple[float, float, float]:
        point_id = int(cls.REF.search(entities[vertex_id][1])[1])
        kind, args = entities[point_id]
        assert kind == "CARTESIAN_POINT", kind
        return tuple(float(c) for c in re.search(r"\(([^()]*)\)", args)[1].split(","))
