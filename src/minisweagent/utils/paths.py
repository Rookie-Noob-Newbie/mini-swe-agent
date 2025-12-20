from pathlib import Path


def get_repo_root() -> Path:
    """Return repository root (assumes src/minisweagent/utils/...)."""
    return Path(__file__).resolve().parents[3]


def get_repo_tmp() -> Path:
    """Return repo-local tmp directory, creating it if needed."""
    tmp = get_repo_root() / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp
