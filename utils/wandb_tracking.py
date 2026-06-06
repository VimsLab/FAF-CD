import os
import subprocess
from typing import Any, Dict


def _safe_git_output(args, cwd):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def collect_git_metadata(cwd=None):
    cwd = cwd or os.getcwd()
    meta = {
        "available": False,
        "commit": None,
        "commit_short": None,
        "branch": None,
        "remote": None,
        "dirty": None,
    }

    commit = _safe_git_output(["rev-parse", "HEAD"], cwd)
    if not commit:
        return meta

    meta["available"] = True
    meta["commit"] = commit
    meta["commit_short"] = _safe_git_output(["rev-parse", "--short", "HEAD"], cwd)
    meta["branch"] = _safe_git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    meta["remote"] = _safe_git_output(["config", "--get", "remote.origin.url"], cwd)

    status = _safe_git_output(["status", "--porcelain"], cwd)
    if status is not None:
        meta["dirty"] = bool(status)

    return meta


def inject_git_metadata_to_wandb_run(wandb_run, git_meta: Dict[str, Any], key_prefix: str = "git"):
    if wandb_run is None or not isinstance(git_meta, dict):
        return

    try:
        wandb_run.config.update({key_prefix: git_meta}, allow_val_change=True)
    except Exception:
        pass
