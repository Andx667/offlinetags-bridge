"""Where the daemon keeps its state and finds its models. Standard library only."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "offlinetags-bridge"
TAG_MODEL = "tags.onnx"
PERSON_MODEL = "person.onnx"


def state_dir() -> Path:
    """Per-user directory for the daemon's state file, log and default models.

    Deliberately not under %LOCALAPPDATA%: Microsoft Store builds of Python
    silently redirect writes there, so the paths we report would not exist.
    """
    return Path.home() / f".{APP_NAME}"


def state_file() -> Path:
    return state_dir() / "daemon.json"


def log_file() -> Path:
    return state_dir() / "daemon.log"


def models_dir() -> Path:
    """Resolve the model directory.

    Priority: OFFLINETAGS_BRIDGE_MODELS, then ``models/`` in a source checkout,
    then the per-user state directory.
    """
    env = os.environ.get("OFFLINETAGS_BRIDGE_MODELS")
    if env:
        return Path(env)
    checkout = Path(__file__).resolve().parents[2] / "models"
    if (checkout / TAG_MODEL).exists():
        return checkout
    return state_dir() / "models"
