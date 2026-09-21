from pathlib import Path, PurePosixPath
import textwrap

import pytest

from bmd_agent.config import (
    CONFIG_ENV_VAR,
    DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS,
    DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
    ConfigurationError,
    load_resources,
    parse_resources,
    resolve_resources_path,
    user_config_path,
)


def valid_config() -> dict:
    return {
        "repositories": {
            "bmd_compute": {
                "name": "BMD Compute",
                "path": "/home/example/projects/bmd_compute",
                "role": "compute",
                "access": "read_only",
                "protected": True,
                "live": True,
                "capability_python": "/home/example/micromamba/envs/bmd-compute/bin/python",
            }
        },
        "clusters": {
            "powerslurm": {
                "name": "PowerSLURM",
                "ssh_host": "powerslurm-bmdguest",
                "partition": "leeburton-pool",
                "access": "observational",
                "allowed_remote_roots": ["/home/example/calculations"],
            }
        },
    }


def test_resolve_resources_path_prefers_environment_variable() -> None:
    path = resolve_resources_path(environ={CONFIG_ENV_VAR: "~/bmd/resources.toml"})

    assert path == Path("~/bmd/resources.toml").expanduser()


def test_user_config_path_uses_windows_appdata() -> None:
    path = user_config_path(
        platform_name="win32",
        environ={"APPDATA": r"C:\Users\Ada\AppData\Roaming"},
        home=Path(r"C:\Users\Ada"),
    )

    assert path == Path(r"C:\Users\Ada\AppData\Roaming") / "bmd-agent" / "resources.toml"


def test_user_config_path_uses_linux_xdg_default() -> None:
    path = user_config_path(platform_name="linux", environ={}, home=Path("/home/ada"))

    assert path == Path("/home/ada/.config/bmd-agent/resources.toml")


def test_parse_resources_returns_typed_resources() -> None:
    registry = parse_resources(valid_config())

    assert registry.repositories["bmd_compute"].access == "read_only"
    assert registry.repositories["bmd_compute"].protected is True
    assert registry.repositories["bmd_compute"].capability_python is not None
    assert registry.repositories["bmd_compute"].capability_python.parts[-3:] == (
        "bmd-compute",
        "bin",
        "python",
    )
    assert registry.clusters["powerslurm"].partition == "leeburton-pool"
    assert registry.clusters["powerslurm"].allowed_remote_roots == (
        PurePosixPath("/home/example/calculations"),
    )
    assert registry.clusters["powerslurm"].deployment_profile is None
    assert (
        registry.clusters["powerslurm"].ssh_connect_timeout_seconds
        == DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS
    )
    assert (
        registry.clusters["powerslurm"].remote_command_timeout_seconds
        == DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS
    )
    assert (
        registry.clusters["powerslurm"].scheduler_accounting_timeout_seconds
        == DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS
    )


def test_parse_resources_preserves_profile_and_operational_timeouts() -> None:
    config = valid_config()
    cluster = config["clusters"]["powerslurm"]
    cluster.update(
        {
            "deployment_profile": "power",
            "ssh_connect_timeout_seconds": 12,
            "remote_command_timeout_seconds": 30,
            "scheduler_accounting_timeout_seconds": 90,
        }
    )

    parsed = parse_resources(config).clusters["powerslurm"]

    assert parsed.deployment_profile == "power"
    assert parsed.ssh_connect_timeout_seconds == 12
    assert parsed.remote_command_timeout_seconds == 30
    assert parsed.scheduler_accounting_timeout_seconds == 90


def test_parse_resources_rejects_mutating_repository_access() -> None:
    config = valid_config()
    config["repositories"]["bmd_compute"]["access"] = "write"

    with pytest.raises(ConfigurationError, match="read_only"):
        parse_resources(config)


def test_parse_resources_rejects_unsafe_partition() -> None:
    config = valid_config()
    config["clusters"]["powerslurm"]["partition"] = "pool;scancel 1"

    with pytest.raises(ConfigurationError, match="unsafe"):
        parse_resources(config)


def test_parse_resources_requires_allowed_remote_roots() -> None:
    config = valid_config()
    config["clusters"]["powerslurm"]["allowed_remote_roots"] = []

    with pytest.raises(ConfigurationError, match="must not be empty"):
        parse_resources(config)


@pytest.mark.parametrize(
    "field",
    (
        "ssh_connect_timeout_seconds",
        "remote_command_timeout_seconds",
        "scheduler_accounting_timeout_seconds",
    ),
)
def test_parse_resources_rejects_nonpositive_operational_timeouts(field: str) -> None:
    config = valid_config()
    config["clusters"]["powerslurm"][field] = 0

    with pytest.raises(ConfigurationError, match="positive integer"):
        parse_resources(config)


def test_parse_resources_rejects_unsafe_deployment_profile_id() -> None:
    config = valid_config()
    config["clusters"]["powerslurm"]["deployment_profile"] = "power;id"

    with pytest.raises(ConfigurationError, match="deployment_profile is invalid"):
        parse_resources(config)


def test_load_resources_reports_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "resources.toml"

    with pytest.raises(ConfigurationError) as exc_info:
        load_resources(missing)

    message = str(exc_info.value)
    assert str(missing) in message
    assert "Create one from" in message
    assert "BMD_AGENT_RESOURCES" in message
    assert "not used for real execution" in message


def test_load_resources_reads_toml_file(tmp_path: Path) -> None:
    config_file = tmp_path / "resources.toml"
    config_file.write_text(
        textwrap.dedent(
            """
            [repositories.bmd_compute]
            name = "BMD Compute"
            path = "/home/example/projects/bmd_compute"
            role = "compute"
            access = "read_only"
            protected = true
            live = true

            [clusters.powerslurm]
            name = "PowerSLURM"
            ssh_host = "powerslurm-bmdguest"
            partition = "leeburton-pool"
            access = "observational"
            allowed_remote_roots = ["/home/example/calculations"]
            """
        ),
        encoding="utf-8",
    )

    registry = load_resources(config_file)

    assert registry.repositories["bmd_compute"].name == "BMD Compute"
