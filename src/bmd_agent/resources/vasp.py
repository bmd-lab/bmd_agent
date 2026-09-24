from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
import subprocess
from typing import Callable, Iterable

from pymatgen.io.vasp import Poscar


Runner = Callable[..., subprocess.CompletedProcess[bytes]]

_ARCHIVE_PROBE_MARKER = "bmd-agent-archive-probe-v1"
_MAX_ARCHIVE_PROBE_LIMIT = 64

_OUTCAR_FORCE_EXTRACTOR_AWK_EMIT = (
    'complete=(emit_status=="complete"&&rows>0&&malformed==0);'
    'final_status=emit_status;value="";'
    'if(complete&&expected!=""&&rows!=expected+0){complete=0;final_status="row_count_mismatch"}'
    'if(malformed){complete=0;final_status="malformed"}'
    'if(complete){value=max_force}'
    'printf("block\\t%d\\t%d\\t%s\\t%d\\t%s\\n",block_no,rows,final_status,complete,value);'
    'state=0;rows=0;max_force=0;malformed=0'
)

_OUTCAR_FORCE_EXTRACTOR_AWK = (
    'BEGIN{print "schema\\tbmd-agent-outcar-force-v1";'
    'print "expected_site_count\\t" expected;'
    'num="^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([Ee][-+]?[0-9]+)?$"}'
    '/^[[:space:]]*POSITION[[:space:]]+TOTAL-FORCE[[:space:]]+\\(eV\\/Angst\\)[[:space:]]*$/{'
    'if(state){emit_status="incomplete";'
    + _OUTCAR_FORCE_EXTRACTOR_AWK_EMIT
    + '};block_no++;state=1;rows=0;max_force=0;malformed=0;next}'
    'state&&/^[[:space:]]*---[-]*[[:space:]]*$/{'
    'if(state==1){state=2}else{emit_status="complete";'
    + _OUTCAR_FORCE_EXTRACTOR_AWK_EMIT
    + '};next}'
    'state&&NF{'
    'if(state==1){state=2;malformed=1}'
    'if(NF<6||$4!~num||$5!~num||$6!~num){malformed=1;next}'
    'fx=$4+0;fy=$5+0;fz=$6+0;rows++;force=sqrt(fx*fx+fy*fy+fz*fz);'
    'if(force>max_force){max_force=force}next}'
    'END{if(state){emit_status="incomplete";'
    + _OUTCAR_FORCE_EXTRACTOR_AWK_EMIT
    + '}}'
)


class RemoteOutcarForceExtractionError(RuntimeError):
    """Raised when the fixed remote OUTCAR extractor fails before valid output."""

    def __init__(
        self,
        public_message: str,
        *,
        kind: str,
        returncode: int | None = None,
        stderr_summary: str | None = None,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.kind = kind
        self.returncode = returncode
        self.stderr_summary = stderr_summary


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


def retrieve_remote_file_tail(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    limit: int,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> bytes:
    """Retrieve a bounded tail from one already-authorized remote file."""

    if limit <= 0:
        raise ValueError("remote tail byte limit must be positive")
    remote_command = " ".join(
        ["tail", "-c", str(limit), "--", shlex.quote(str(remote_path))]
    )
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


def probe_remote_error_archives(
    ssh_host: str,
    directory: PurePosixPath | str,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
    limit: int,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> tuple[PurePosixPath, ...]:
    """Probe a bounded contiguous error.N.tar.gz sequence without reading archives."""

    if limit <= 0 or limit > _MAX_ARCHIVE_PROBE_LIMIT:
        raise ValueError(
            f"remote error archive probe limit must be between 1 and {_MAX_ARCHIVE_PROBE_LIMIT}"
        )
    authorized_directory = authorize_remote_path(
        directory,
        allowed_roots=allowed_roots,
    )
    program = (
        'i=1; while [ "$i" -le "$2" ]; do '
        'candidate="$1/error.$i.tar.gz"; '
        'if [ -f "$candidate" ]; then printf "%s\\n" "$candidate"; else break; fi; '
        'i=$((i + 1)); done'
    )
    remote_command = " ".join(
        (
            "sh",
            "-c",
            shlex.quote(program),
            _ARCHIVE_PROBE_MARKER,
            shlex.quote(str(authorized_directory)),
            str(limit),
        )
    )
    result = runner(
        ["ssh", ssh_host, remote_command],
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        return ()

    lines = result.stdout.decode("utf-8", "strict").splitlines()
    if len(lines) > limit:
        raise ValueError("remote error archive probe returned too many paths")
    archives: list[PurePosixPath] = []
    for index, line in enumerate(lines, start=1):
        expected = authorized_directory / f"error.{index}.tar.gz"
        if line != str(expected):
            raise ValueError("remote error archive probe returned an unexpected path")
        archives.append(expected)
    return tuple(archives)


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
    try:
        result = runner(
            ["ssh", ssh_host, remote_command],
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise _outcar_extraction_error(exc) from exc

    return result.stdout.decode("utf-8", "replace")


def build_remote_outcar_force_command(
    remote_path: PurePosixPath,
    *,
    expected_site_count: int | None = None,
) -> str:
    """Build the fixed read-only remote OUTCAR force extractor command."""

    expected = "" if expected_site_count is None else str(expected_site_count)
    command = [
        "awk",
        "-v",
        "expected=" + expected,
        shlex.quote(_OUTCAR_FORCE_EXTRACTOR_AWK),
        shlex.quote(str(remote_path)),
    ]
    if expected_site_count is not None:
        if expected_site_count <= 0:
            raise ValueError("expected site count must be positive")

    return " ".join(command)


def _outcar_extraction_error(
    exc: subprocess.CalledProcessError,
) -> RemoteOutcarForceExtractionError:
    stderr_excerpt = _stderr_excerpt(exc.stderr)
    stderr_summary = _stderr_summary(stderr_excerpt)
    lower = (stderr_excerpt or "").lower()
    kind = "remote_command_failed"
    message = "extractor command failed"

    if (
        "cannot open" in lower
        or "permission denied" in lower
        or "no such file" in lower
    ):
        kind = "outcar_read_failed"
        message = "file could not be read"
    elif exc.returncode == 127 or "command not found" in lower or "awk: not found" in lower:
        kind = "extractor_runtime_unavailable"
        message = "remote extractor runtime unavailable"
    elif (
        "syntax error" in lower
        or "unexpected eof" in lower
        or "unterminated" in lower
        or "unexpected token" in lower
    ):
        kind = "command_invocation_failed"
        message = "extractor command invocation failed"

    return RemoteOutcarForceExtractionError(
        message,
        kind=kind,
        returncode=exc.returncode,
        stderr_summary=stderr_summary,
    )


def _stderr_summary(stderr: bytes | str | None) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return None


def _stderr_excerpt(stderr: bytes | str | None) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        if len(lines) >= 12:
            break
    excerpt = "\n".join(lines)
    return excerpt[:2000] if excerpt else None


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
