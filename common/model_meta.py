"""Metadata saved next to each trained model: when it was trained, how it scored,
and with which library versions (saved models are tied to those versions)."""

import json
import logging
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

TRACKED_PACKAGES = ("scikit-learn", "torch", "transformers")

logger = logging.getLogger(__name__)


def _installed_versions():
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _release(version_string):
    """Drop the local build label, so torch "2.14.1+cpu" and "2.14.1+cu130" compare equal."""
    return version_string.split("+", 1)[0]


def write_metadata(path, metrics, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": metrics,
        "library_versions": _installed_versions(),
        **extra,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")


def read_metadata(path, model_name, log=logger):
    """Return the saved metadata (or None), logging a warning if library versions changed since training."""
    try:
        meta = json.loads(path.read_text())
    except FileNotFoundError:
        log.warning("No metadata at %s; can't verify which library versions trained the %s model", path, model_name)
        return None
    except (OSError, ValueError):
        log.exception("Could not read model metadata at %s", path)
        return None

    installed = _installed_versions()
    for package, trained_with in meta.get("library_versions", {}).items():
        if trained_with and installed.get(package) and _release(installed[package]) != _release(trained_with):
            log.warning(
                "%s model was trained with %s %s but %s is installed; retrain if predictions look wrong",
                model_name,
                package,
                trained_with,
                installed[package],
            )
    return meta
