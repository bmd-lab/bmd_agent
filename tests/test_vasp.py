import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess

import pytest

from bmd_agent.resources.vasp import (
    RemotePathError,
    RemoteOutcarForceExtractionError,
    authorize_remote_path,
    build_remote_outcar_force_command,
    build_remote_file_path,
    extract_remote_outcar_force_blocks,
    parse_poscar,
    read_remote_structure,
    remote_directory_exists,
    remote_file_exists,
    remote_file_size,
    retrieve_remote_file,
)


POSCAR = """\
Si
1.0
5.43 0.00 0.00
0.00 5.43 0.00
0.00 0.00 5.43
Si
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
"""


def test_parse_poscar_fixture() -> None:
    info = parse_poscar(POSCAR, source="/remote/POSCAR")

    assert info.source == "/remote/POSCAR"
    assert info.reduced_formula == "Si"
    assert info.sites == 2
    assert info.a == pytest.approx(5.43)
    assert info.b == pytest.approx(5.43)
    assert info.c == pytest.approx(5.43)


def test_remote_posix_path_construction_is_local_os_independent() -> None:
    path = build_remote_file_path(
        "/home/example/calculations/project",
        "POSCAR",
        allowed_roots=["/home/example/calculations"],
    )

    assert path == PurePosixPath("/home/example/calculations/project/POSCAR")
    assert "\\" not in str(path)


def test_remote_path_rejects_outside_allowed_roots() -> None:
    with pytest.raises(RemotePathError, match="outside configured allowed roots"):
        build_remote_file_path(
            "/home/example/calculations/../other",
            "POSCAR",
            allowed_roots=["/home/example/calculations"],
        )


def test_remote_path_rejects_relative_directory() -> None:
    with pytest.raises(RemotePathError, match="absolute"):
        build_remote_file_path(
            "relative/project",
            "POSCAR",
            allowed_roots=["/home/example/calculations"],
        )


def test_authorize_remote_path_rechecks_producer_supplied_absolute_path() -> None:
    path = authorize_remote_path(
        "/home/example/calculations/run/submission.json",
        allowed_roots=["/home/example/calculations"],
    )

    assert path == PurePosixPath("/home/example/calculations/run/submission.json")


def test_authorize_remote_path_rejects_producer_supplied_path_outside_roots() -> None:
    with pytest.raises(RemotePathError, match="outside configured allowed roots"):
        authorize_remote_path(
            "/home/example/other/run/submission.json",
            allowed_roots=["/home/example/calculations"],
        )


def test_retrieve_remote_file_uses_quoted_read_only_cat_command() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, stdout=b"data", stderr=b"")

    contents = retrieve_remote_file(
        "powerslurm-bmdguest",
        PurePosixPath("/home/example/calculations/project with spaces/POSCAR"),
        runner=runner,
    )

    assert contents == b"data"
    assert calls == [
        [
            "ssh",
            "powerslurm-bmdguest",
            "cat -- '/home/example/calculations/project with spaces/POSCAR'",
        ]
    ]


def test_remote_exists_helpers_use_read_only_test_commands() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    assert remote_file_exists(
        "powerslurm-bmdguest",
        PurePosixPath("/home/example/calculations/run/vasprun.xml"),
        runner=runner,
    )
    assert remote_directory_exists(
        "powerslurm-bmdguest",
        PurePosixPath("/home/example/calculations/run"),
        runner=runner,
    )

    assert calls == [
        [
            "ssh",
            "powerslurm-bmdguest",
            "test -f /home/example/calculations/run/vasprun.xml",
        ],
        [
            "ssh",
            "powerslurm-bmdguest",
            "test -d /home/example/calculations/run",
        ],
    ]


def test_remote_file_size_uses_read_only_stat_command() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, stdout=b"123\n", stderr=b"")

    size = remote_file_size(
        "powerslurm-bmdguest",
        PurePosixPath("/home/example/calculations/project with spaces/vasprun.xml"),
        runner=runner,
    )

    assert size == 123
    assert calls == [
        [
            "ssh",
            "powerslurm-bmdguest",
            "stat -c %s -- '/home/example/calculations/project with spaces/vasprun.xml'",
        ]
    ]


def test_remote_outcar_force_extractor_uses_fixed_read_only_awk_command() -> None:
    command = build_remote_outcar_force_command(
        PurePosixPath("/home/example/calculations/project with spaces/OUTCAR"),
        expected_site_count=24,
    )

    parts = command.split(" ")
    assert parts[0:3] == ["awk", "-v", "expected=24"]
    assert "\n" not in command
    assert "python3" not in command
    assert "function " not in command
    assert "index" not in command
    assert "block_no" in command
    assert "find " not in command
    assert "ls " not in command
    assert "cat " not in command
    assert "POTCAR" not in command
    assert "'/home/example/calculations/project with spaces/OUTCAR'" in command


def test_remote_outcar_force_awk_program_avoids_cluster_rejected_constructs() -> None:
    command = build_remote_outcar_force_command(
        PurePosixPath("/home/example/calculations/run/OUTCAR"),
        expected_site_count=24,
    )

    assert "}function" not in command
    assert "function emit" not in command
    assert "index++" not in command
    assert "\\(eV\\/Angst\\)" in command


def test_remote_outcar_force_awk_program_runs_with_system_awk_when_available(
    tmp_path: Path,
) -> None:
    awk = shutil.which("awk")
    if awk is None or os.name == "nt":
        pytest.skip("system awk is not available in this environment")
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        """\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      3.00000000      4.00000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      1.00000000
 -----------------------------------------------------------------------------------
""",
        encoding="utf-8",
    )
    command = build_remote_outcar_force_command(
        PurePosixPath(outcar.as_posix()),
        expected_site_count=2,
    )

    result = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    assert "schema\tbmd-agent-outcar-force-v1" in result.stdout
    assert "block\t1\t2\tcomplete\t1\t5" in result.stdout


def test_extract_remote_outcar_force_blocks_returns_compact_stdout() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"schema\tbmd-agent-outcar-force-v1\nexpected_site_count\t2\n",
            stderr=b"",
        )

    payload = extract_remote_outcar_force_blocks(
        "powerslurm-bmdguest",
        PurePosixPath("/home/example/calculations/run/OUTCAR"),
        expected_site_count=2,
        runner=runner,
    )

    assert payload == "schema\tbmd-agent-outcar-force-v1\nexpected_site_count\t2\n"
    assert len(calls) == 1
    assert calls[0][0:2] == ["ssh", "powerslurm-bmdguest"]
    assert calls[0][2].startswith("awk -v expected=2 ")
    assert "\n" not in calls[0][2]
    assert calls[0][2].endswith(" /home/example/calculations/run/OUTCAR")


def test_extract_remote_outcar_force_blocks_classifies_remote_runtime_failure() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(
            127,
            command,
            stderr=b"sh: 1: awk: not found\n",
        )

    with pytest.raises(RemoteOutcarForceExtractionError) as exc_info:
        extract_remote_outcar_force_blocks(
            "powerslurm-bmdguest",
            PurePosixPath("/home/example/calculations/run/OUTCAR"),
            runner=runner,
        )

    assert exc_info.value.kind == "extractor_runtime_unavailable"
    assert exc_info.value.returncode == 127
    assert exc_info.value.public_message == "remote extractor runtime unavailable"


def test_extract_remote_outcar_force_blocks_classifies_invocation_failure() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(
            2,
            command,
            stderr=b"sh: 1: Syntax error: Unterminated quoted string\n",
        )

    with pytest.raises(RemoteOutcarForceExtractionError) as exc_info:
        extract_remote_outcar_force_blocks(
            "powerslurm-bmdguest",
            PurePosixPath("/home/example/calculations/run/OUTCAR"),
            runner=runner,
        )

    assert exc_info.value.kind == "command_invocation_failed"
    assert exc_info.value.public_message == "extractor command invocation failed"


def test_extract_remote_outcar_force_blocks_classifies_multiline_awk_syntax_failure() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr=(
                b"awk: cmd. line:1: BEGIN{print \"schema\"}function emit(){...}\n"
                b"awk: cmd. line:1:                                  ^ syntax error\n"
            ),
        )

    with pytest.raises(RemoteOutcarForceExtractionError) as exc_info:
        extract_remote_outcar_force_blocks(
            "powerslurm-bmdguest",
            PurePosixPath("/home/example/calculations/run/OUTCAR"),
            runner=runner,
        )

    assert exc_info.value.kind == "command_invocation_failed"
    assert exc_info.value.returncode == 1
    assert exc_info.value.public_message == "extractor command invocation failed"


def test_extract_remote_outcar_force_blocks_classifies_outcar_read_failure() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(
            2,
            command,
            stderr=b"awk: cannot open /home/example/calculations/run/OUTCAR (Permission denied)\n",
        )

    with pytest.raises(RemoteOutcarForceExtractionError) as exc_info:
        extract_remote_outcar_force_blocks(
            "powerslurm-bmdguest",
            PurePosixPath("/home/example/calculations/run/OUTCAR"),
            runner=runner,
        )

    assert exc_info.value.kind == "outcar_read_failed"
    assert exc_info.value.public_message == "file could not be read"


def test_read_remote_structure_combines_authorized_retrieval_and_parsing() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=POSCAR.encode("utf-8"), stderr=b"")

    info = read_remote_structure(
        "powerslurm-bmdguest",
        "/home/example/calculations/project",
        allowed_roots=["/home/example/calculations"],
        runner=runner,
    )

    assert info.source == "/home/example/calculations/project/POSCAR"
    assert info.reduced_formula == "Si"
