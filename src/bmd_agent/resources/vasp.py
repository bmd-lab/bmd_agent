from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from pymatgen.io.vasp import Poscar


SSH_HOST = "powerslurm-bmdguest"


@dataclass
class StructureInfo:
    source: str
    formula: str
    reduced_formula: str
    sites: int
    volume: float
    a: float
    b: float
    c: float


def read_remote_structure(
    directory: str,
    filename: str = "POSCAR",
) -> StructureInfo:
    """Read a VASP structure from PowerSLURM without modifying it."""

    remote_path = str(Path(directory) / filename)

    result = subprocess.run(
        ["ssh", SSH_HOST, "cat", remote_path],
        capture_output=True,
        check=True,
    )

    with tempfile.NamedTemporaryFile(mode="wb") as tmp:
        tmp.write(result.stdout)
        tmp.flush()

        poscar = Poscar.from_file(tmp.name)
        structure = poscar.structure

    lattice = structure.lattice

    return StructureInfo(
        source=remote_path,
        formula=structure.composition.formula,
        reduced_formula=structure.composition.reduced_formula,
        sites=len(structure),
        volume=structure.volume,
        a=lattice.a,
        b=lattice.b,
        c=lattice.c,
    )
