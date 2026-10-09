"""Provenance stamped into every artifact, and the schema check on read.

An old artifact and a new one used to be indistinguishable: a curve written before
``k_effective`` existed was read as if it were current, and nothing recorded which
commit, dtype or dependency versions produced a number.  Every ``save`` now stamps
a ``schema_version`` and a ``provenance`` block, and every reader warns when the
stamp is missing or older.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

#: Bump when a stored field changes meaning or a required field is added.
#: Bumped to 5: the causal artifacts now carry the eval-needle text and the masked
#: head/layer audit (and the summary carries recited-only matrices), so a v4 reader
#: would miss the fields that make those runs reproducible.
SCHEMA_VERSION = 5

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_state() -> tuple[str | None, bool]:
    """(HEAD commit, working tree dirty).  ``(None, False)`` outside a checkout.

    Deliberately *not* cached: a long run writes many artifacts, and the dirty
    flag is exactly the field that can change between them (a test or a fix lands
    mid-run).  Two cheap ``git`` calls per artifact are worth the accuracy.
    """
    try:
        rev = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            # `--untracked-files=no`: the repo is full of untracked working files
            # (HANDOFF.md, scratch, pulled logs), so the default made `git_dirty`
            # report True almost always and say nothing.
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        return (rev or None), dirty
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return None, False


@lru_cache(maxsize=1)
def _deterministic() -> bool:
    """Whether deterministic kernels are enabled (argmax ties can flip otherwise)."""
    try:
        import torch

        return bool(torch.are_deterministic_algorithms_enabled()
                    or getattr(torch.backends.cudnn, "deterministic", False))
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def _versions() -> dict[str, str]:
    # Cached: importing matplotlib on every artifact write cost seconds per save.
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "numpy", "matplotlib"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = "?"
    return out


def provenance(*, dtype: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """A dict describing exactly what produced an artifact."""
    rev, dirty = _git_state()
    versions = _versions()
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_rev": rev,
        "git_dirty": dirty,
        "python": versions["python"],
        "torch": versions["torch"],
        "transformers": versions["transformers"],
        "numpy": versions["numpy"],
        "matplotlib": versions["matplotlib"],
        # argmax ties can flip between runs if deterministic kernels are off.
        "deterministic": _deterministic(),
    }
    # A job uploads `local-paths` without `.git`, so `git_rev` is None there and the
    # artifact could not be tied to a commit.  `datasphere_job.py` hashes the code it
    # uploaded and exports it as RH_CODE_SHA256; keep the name explicit so a reader
    # does not mistake it for a git object.
    code_sha = os.environ.get("RH_CODE_SHA256")
    if code_sha:
        data["code_sha256"] = code_sha
    if dtype:
        data["dtype"] = str(dtype)
    if extra:
        data.update(extra)
    return data


def add_provenance(
    payload: dict[str, Any],
    *,
    dtype: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``payload`` with ``schema_version`` and ``provenance`` added."""
    out = dict(payload)
    out["schema_version"] = SCHEMA_VERSION
    out["provenance"] = provenance(dtype=dtype, extra=extra)
    return out


def warn_if_stale(
    payload: dict[str, Any],
    what: str,
    *,
    log: Callable[..., None],
) -> None:
    """Warn when an artifact predates the current schema (or has no stamp)."""
    version = payload.get("schema_version")
    if version is None:
        log.warning("%s has no schema_version (written by an older version of this "
                    "code); fields may be missing and defaults may be wrong", what)
    elif version != SCHEMA_VERSION:
        log.warning("%s has schema_version %s, this code writes %s; re-run the stage",
                    what, version, SCHEMA_VERSION)
