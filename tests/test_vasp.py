from pathlib import PurePosixPath
import subprocess

import pytest

from bmd_agent.resources.vasp import (
    RemotePathError,
    authorize_remote_path,
    build_remote_file_path,
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
