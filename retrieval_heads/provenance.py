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
#: Bumped to 6: three further rounds added fields *and* changed semantics without a
#: bump -- the ablation artifacts gained truncation counters, `control_exhausted` and
#: `random_distinct`; the QA/CoT payloads gained threshold/pairing/argmax_domain; the
#: aligned ranking gained n/n_missing; the control subsets are now drawn without
#: repetition (so a re-run's random arm differs); and the pairing rows are scoped by
#: what was actually generated.  Without the bump a schema-5 artifact from before all
#: of that read as current.
SCHEMA_VERSION = 6

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Set once a `git` call has failed, so a job (which has no `.git` -- `local-paths`
#: uploads the package, not the repository) does not spawn two doomed subprocesses per
#: artifact.  A paper-scale detect writes one JSONL line per instance *and* a summary
#: per pairing, so that was ~1200 process spawns for nothing.
_NO_GIT = False


def _git_state() -> tuple[str | None, bool]:
    """(HEAD commit, working tree dirty).  ``(None, False)`` outside a checkout.

    Deliberately *not* cached on success: a long run writes many artifacts, and the
    dirty flag is exactly the field that can change between them (a test or a fix
    lands mid-run).  Two cheap ``git`` calls per artifact are worth that accuracy --
    but only while `git` works at all.
    """
    global _NO_GIT
    if _NO_GIT:
        return None, False
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
        # Remember it: in a job this fails every single time, and the artifacts
        # already say `git_rev: null` (plus `code_sha256`, set by the driver).
        _NO_GIT = True
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
def _optional_kernels() -> dict[str, bool]:
    """Whether the fused kernels are *installed* (not whether they were used).

    `transformers` imports them lazily and falls back to pure PyTorch when the
    import fails, so two runs of the same code can use different kernels.  The
    packages are named, not imported: `find_spec` is cheap and side-effect free, and
    a CUDA extension that imports is not necessarily one that runs.
    """
    import importlib.util

    out: dict[str, bool] = {}
    for label, module in (("flash_linear_attention", "fla"),
                          ("causal_conv1d", "causal_conv1d")):
        try:
            out[label] = importlib.util.find_spec(module) is not None
        except Exception:  # noqa: BLE001 - provenance must never break a run
            out[label] = False
    return out


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
        # Which optional kernels were *available*: the hybrid's linear layers take a
        # different (fused) path when these are installed, so an artifact from a run
        # with them is not bit-for-bit the same experiment as one without.
        "optional_kernels": _optional_kernels(),
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
