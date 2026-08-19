from pathlib import Path
import tomllib


def project_root() -> Path:
    """Return the root directory of the bmd_agent source checkout."""
    return Path(__file__).resolve().parents[2]


def load_resources() -> dict:
    """Load configured BMD resources."""
    config_path = project_root() / "config" / "resources.toml"

    with config_path.open("rb") as handle:
        return tomllib.load(handle)
