from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
import subprocess
from typing import Callable, Iterable

from pymatgen.io.vasp import Poscar


Runner = Callable[..., subprocess.CompletedProcess[bytes]]

_OUTCAR_FORCE_EXTRACTOR_SCRIPT = r"""
import json
import math
import re
import sys

HEADER = re.compile(r"^\s*POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)\s*$")
SEPARATOR = re.compile(r"^\s*-{3,}\s*$")

args = sys.argv[1:]
if args and args[0] == "--":
    args = args[1:]
path = args[0]
expected = int(args[1]) if len(args) > 1 else None
blocks = []
state = None
index = 0
rows = 0
max_force = 0.0
malformed = False


def finish(status):
    global state, rows, max_force, malformed
    complete = status == "complete" and rows > 0 and not malformed
    final_status = status
    value = max_force if complete else None
    if complete and expected is not None and rows != expected:
        complete = False
        final_status = "row_count_mismatch"
        value = None
    if malformed:
        complete = False
        final_status = "malformed"
        value = None
    blocks.append(
        {
            "block_index": index,
            "row_count": rows,
            "status": final_status,
            "complete": complete,
            "max_force_eV_per_A": value,
        }
    )
    state = None
    rows = 0
    max_force = 0.0
    malformed = False


with open(path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if HEADER.match(line):
            if state is not None:
                finish("incomplete")
            index += 1
            state = "await_separator"
            rows = 0
            max_force = 0.0
            malformed = False
            continue

        if state is None:
            continue

        if SEPARATOR.match(line):
            if state == "await_separator":
                state = "rows"
            else:
                finish("complete")
            continue

        if not line.strip():
            continue

        if state == "await_separator":
            state = "rows"
            malformed = True

        parts = line.split()
        if len(parts) < 6:
            malformed = True
            continue
        try:
            fx, fy, fz = (float(value) for value in parts[3:6])
        except Exception:
            malformed = True
            continue
        rows += 1
        max_force = max(max_force, math.sqrt(fx * fx + fy * fy + fz * fz))

if state is not None:
    finish("incomplete")

print(
    json.dumps(
        {
            "schema": "bmd-agent-outcar-force-v1",
            "expected_site_count": expected,
            "blocks": blocks,
        },
        separators=(",", ":"),
    )
)
""".strip()


class RemotePathError(ValueError):
    """Raised when a requested remote path is outside configured policy."""


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
    ssh_host: str,
    directory: str,
    allowed_roots: Iterable[PurePosixPath | str],
    filename: str = "POSCAR",
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> StructureInfo:
    """Read a VASP structure remotely without modifying the source."""

    remote_path = build_remote_file_path(
        directory,
        filename,
        allowed_roots=allowed_roots,
    )
    contents = retrieve_remote_file(
        ssh_host,
        remote_path,
        runner=runner,
        timeout=timeout,
    )

    return parse_poscar(contents, source=str(remote_path))


def retrieve_remote_file(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> bytes:
    """Retrieve one already-authorized remote file over SSH."""

    remote_command = "cat -- " + shlex.quote(str(remote_path))

    result = runner(
        ["ssh", ssh_host, remote_command],
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    return result.stdout


def remote_file_exists(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> bool:
    """Return whether one already-authorized remote file exists."""

    return _remote_path_exists(
        ssh_host,
        remote_path,
        test_flag="-f",
        runner=runner,
        timeout=timeout,
    )


def remote_file_size(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> int:
    """Return the byte size of one already-authorized remote file."""

    remote_command = "stat -c %s -- " + shlex.quote(str(remote_path))
    result = runner(
        ["ssh", ssh_host, remote_command],
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    return int(result.stdout.decode("utf-8", "replace").strip())


def extract_remote_outcar_force_blocks(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    expected_site_count: int | None = None,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> str:
    """Return compact OUTCAR force-block evidence without transferring the OUTCAR."""

    remote_command = build_remote_outcar_force_command(
        remote_path,
        expected_site_count=expected_site_count,
    )
    result = runner(
        ["ssh", ssh_host, remote_command],
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    return result.stdout.decode("utf-8", "replace")


def build_remote_outcar_force_command(
    remote_path: PurePosixPath,
    *,
    expected_site_count: int | None = None,
) -> str:
    """Build the fixed read-only remote OUTCAR force extractor command."""

    command = [
        "python3",
        "-c",
        shlex.quote(_OUTCAR_FORCE_EXTRACTOR_SCRIPT),
        "--",
        shlex.quote(str(remote_path)),
    ]
    if expected_site_count is not None:
        if expected_site_count <= 0:
            raise ValueError("expected site count must be positive")
        command.append(str(expected_site_count))

    return " ".join(command)


def remote_directory_exists(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> bool:
    """Return whether one already-authorized remote directory exists."""

    return _remote_path_exists(
        ssh_host,
        remote_path,
        test_flag="-d",
        runner=runner,
        timeout=timeout,
    )


def _remote_path_exists(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    test_flag: str,
    runner: Runner,
    timeout: float,
) -> bool:
    remote_command = "test " + test_flag + " " + shlex.quote(str(remote_path))
    result = runner(
        ["ssh", ssh_host, remote_command],
        capture_output=True,
        check=False,
        timeout=timeout,
    )

    return result.returncode == 0


def parse_poscar(contents: bytes | str, *, source: str) -> StructureInfo:
    """Parse VASP POSCAR content using pymatgen."""

    text = contents.decode("utf-8") if isinstance(contents, bytes) else contents
    poscar = Poscar.from_str(text)
    structure = poscar.structure

    lattice = structure.lattice

    return StructureInfo(
        source=source,
        formula=structure.composition.formula,
        reduced_formula=structure.composition.reduced_formula,
        sites=len(structure),
        volume=structure.volume,
        a=lattice.a,
        b=lattice.b,
        c=lattice.c,
    )


def build_remote_file_path(
    directory: str | PurePosixPath,
    filename: str,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath:
    """Build and authorize a POSIX remote file path."""

    remote_directory = normalize_remote_path(directory)
    remote_filename = PurePosixPath(filename)

    if (
        remote_filename.is_absolute()
        or not remote_filename.parts
        or remote_filename.parts == (".",)
        or ".." in remote_filename.parts
    ):
        raise RemotePathError("remote filename must be relative and must not contain '..'")

    return authorize_remote_path(
        remote_directory / remote_filename,
        allowed_roots=allowed_roots,
    )


def authorize_remote_path(
    path: str | PurePosixPath,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath:
    """Authorize one absolute POSIX remote path against configured roots."""

    remote_path = normalize_remote_path(path)
    normalized_roots = tuple(normalize_remote_path(root) for root in allowed_roots)

    if not normalized_roots:
        raise RemotePathError("no allowed remote roots are configured")

    if not any(is_relative_to(remote_path, root) for root in normalized_roots):
        raise RemotePathError("remote path is outside configured allowed roots")

    return remote_path


def normalize_remote_path(path: str | PurePosixPath) -> PurePosixPath:
    """Lexically normalize an absolute POSIX remote path."""

    raw_path = PurePosixPath(str(path).replace("\\", "/"))

    if not raw_path.is_absolute():
        raise RemotePathError("remote path must be absolute")

    parts: list[str] = []

    for part in raw_path.parts:
        if part in ("", "/", "."):
            continue

        if part == "..":
            if not parts:
                raise RemotePathError("remote path escapes the filesystem root")

            parts.pop()
            continue

        parts.append(part)

    if not parts:
        return PurePosixPath("/")

    return PurePosixPath("/" + "/".join(parts))


def is_relative_to(path: PurePosixPath, root: PurePosixPath) -> bool:
    """Return whether path is inside root using POSIX path parts."""

    return path == root or path.parts[: len(root.parts)] == root.parts
