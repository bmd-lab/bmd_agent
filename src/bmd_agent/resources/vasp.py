from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
import subprocess
from typing import Callable, Iterable

from pymatgen.io.vasp import Poscar


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


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
