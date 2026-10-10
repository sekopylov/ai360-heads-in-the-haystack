#!/usr/bin/env python3
"""DataSphere job entry point for the retrieval-heads reproduction.

Runs **inside the job VM**, not on the laptop.  The job starts in ``/job``;
``retrieval_heads/``, ``configs/``, ``scripts/`` and this file are unpacked there
by the ``local-paths`` section of the job config, so ``REPO_ROOT`` (derived from
the package location) and the ``configs/models.json`` registry both resolve with
no shell glue.

Typical use, from the job config's ``cmd``::

    python3 datasphere_job.py --weights ${WEIGHTS} --models qwen3.5-0.8b qwen3-0.6b \
        --profile laptop --stages describe,detect,mask,qa,cot,compare,figures

Stage scales mirror ``scripts/reproduce_laptop.sh`` and
``scripts/reproduce_gpu.sh`` so a job run stays comparable with a local one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

# Make the uploaded package importable before anything imports it.  A job uploads
# `retrieval_heads/` and `scripts/` side by side into `/job`, but Python puts the
# *script's* directory (`/job/scripts`) on `sys.path[0]`, not the repo root -- and the
# stages run in-process (`run_cli` -> `from retrieval_heads import cli`) since the
# model-resident change.  The per-stage `python -m retrieval_heads.cli` subprocess it
# replaced got `/job` on the path for free, because `-m` adds the cwd.  Without this the
# first stage of every job dies with `ModuleNotFoundError` after the venv, hash and GPU
# checks have already passed (the first A100 preflight did exactly that).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Per-stage flags for each scale, kept in step with the two reproduce scripts.
#:
#: Head counts are given as *fractions* (``--k-frac``) rather than absolute K,
#: because K=8 is 17% of Qwen3.5-0.8B's 48 scoreable heads but only 1.8% of
#: Qwen3-0.6B's 448 -- absolute K silently makes the two models incomparable.
SCALES: dict[str, dict[str, list[str]]] = {
    "smoke": {
        "detect": ["--profile", "smoke", "--argmax-domain", "haystack"],
        "mask": ["--k-frac", "0.04", "0.17", "--lengths", "1024", "--random-trials", "1"],
        "qa": ["--k-frac", "0.08", "--random-trials", "1"],
        "cot": ["--k-frac", "0.08", "--random-trials", "1", "--max-new-tokens", "128"],
    },
    "laptop": {
        "detect": ["--profile", "laptop", "--argmax-domain", "haystack"],
        "mask": ["--k-frac", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "1024", "--random-trials", "2"],
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "2"],
        "cot": ["--k-frac", "0.08", "--random-trials", "1", "--max-new-tokens", "192"],
    },
    #: Single T4 (gt4.1 / gt4i.1, 16-24 GB VRAM).  Much larger than `laptop`,
    #: deliberately smaller than `paper`: the mask stage re-runs the prefill for
    #: every configuration, so its cost grows with contexts x K values x trials.
    "t4": {
        "detect": ["--profile", "paper", "--argmax-domain", "haystack",
                   "--lengths", "1024", "2048", "4096", "8192", "16384",
                   "--depths", "5", "--needles", "3"],
        "mask": ["--k-frac", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "4096", "8192", "--random-trials", "3"],
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "3"],
        "cot": ["--k-frac", "0.08", "--random-trials", "2", "--max-new-tokens", "256"],
    },
    "paper": {
        # The paper's full recipe is 3 needles x 20 lengths in 1K-50K x 10 depths
        # (~600 instances).  Here 9 geometric lengths cover the same span in a
        # fraction of the time; `--profile paper` alone would use 7.
        "detect": ["--profile", "paper", "--argmax-domain", "haystack",
                   "--lengths", "1024", "2048", "4096", "8192", "16384",
                   "24576", "32768", "40960", "49152"],
        "mask": ["--k-frac", "0.01", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "4096", "8192", "16384", "--random-trials", "5"],
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "5"],
        "cot": ["--k-frac", "0.08", "--random-trials", "3", "--max-new-tokens", "256"],
    },
    #: Single A100 (g2.1, 80 GB).  Same grid as `paper` -- so the numbers stay
    #: comparable -- but with a much larger prefill chunk (`--prefill-chunk 8192`
    #: against the 4096 default).  Chunking does not repeat layer work (each token
    #: belongs to one chunk); it only bounds the attention score matrix at
    #: O(chunk x seq), which matters because a float32 SDPA fallback materialises
    #: (heads, seq, seq) and OOM'd a 22 GiB card at 16K (findings section 18).  At 49K
    #: and a 8192 chunk that is 1.61 GB per Q-head -- ~12.9 GB for the hybrid's 8 and
    #: ~25.8 GB for the dense model's 16 -- so the protection stays (80 GB fits it,
    #: 22 does not) while the number of chunks halves.  One-shot (`0`) was the
    #: first version of this profile and it was a bad trade: it saves ~4-8% of the
    #: attention time and gives up the bound entirely.
    "a100": {
        "detect": ["--profile", "paper", "--argmax-domain", "haystack",
                   "--prefill-chunk", "8192", "--max-new-tokens", "96",
                   # Build every planned prompt (CPU only) before the first forward
                   # pass: one prompt whose haystack cannot be located verbatim would
                   # otherwise abort the stage hours into the run.  Per model: 270 for
                   # the hybrid, 210 for the dense model, whose 40960 window drops the
                   # two longest lengths.  The `paper`/`t4` scales leave it off -- there
                   # a mid-grid failure costs minutes.
                   "--preflight",
                   "--lengths", "1024", "2048", "4096", "8192", "16384",
                   "24576", "32768", "40960", "49152"],
        "mask": ["--k-frac", "0.01", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "4096", "8192", "16384", "--random-trials", "5",
                 # Same generation budget as detect: the stage that carries the causal
                 # claim must not measure a drop from a truncated baseline.  The CLI
                 # default is 32 -- tighter than the 48 the committed t4 tree used --
                 # and the hybrid's answers were being cut mid-sentence at 32.
                 "--max-new-tokens", "96",
                 # The detection grid uses 10 depths; the ablation default is 5, which
                 # is the weakest part of the causal measurement (`retrieval_std` is the
                 # spread over exactly these samples).  The sample count is
                 # lengths x depths x needles -- 3 x 5 x 3 = 45 per point here, against
                 # the t4 run's 2 x 5 x 1 = 10 -- so this is 4.5x the t4 budget, not the
                 # "same 15 samples" an earlier comment claimed (that counted only
                 # depths x needles and ignored the three lengths).  Needle identity is
                 # the larger source of variance, so the budget goes to three
                 # (question, needle) pairs rather than ten depths of one.
                 "--depths", "5", "--needles", "3", "--prefill-chunk", "8192"],
        # `qa` keeps the CLI's 24-token default on purpose: the built-in answers are
        # short spans and its metric is token-F1 over the whole completion, so extra
        # narration costs precision rather than buying recall -- unlike `mask`/`detect`,
        # where the score is a recall over needle tokens.  Every artifact records
        # `max_new_tokens`, so the three budgets in one tree are visible rather than
        # implied.
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "5",
               "--prefill-chunk", "8192"],
        "cot": ["--k-frac", "0.08", "--random-trials", "3", "--max-new-tokens", "256",
                "--prefill-chunk", "8192"],
    },
}

STAGES = ("describe", "detect", "mask", "qa", "cot", "compare", "figures",
          "case-study")
#: Stages whose CLI defines ``--lengths``.  The driver's ``--lengths`` override must
#: reach only these: ``qa``/``cot`` have no grid flag at all, so appending it made
#: argparse exit 2 and failed the stage (no shipped config hit the combination, which
#: is why it survived -- ``a100-resume.yaml`` invites adding ``qa``/``cot``).
LENGTH_STAGES = ("detect", "mask")
#: Stages that need a loaded model (and therefore benefit from the one-model cache).
MODEL_STAGES = ("describe", "detect", "mask", "qa", "cot", "case-study")
#: Stages that read every model's artifacts and need no model at all.
MODEL_FREE_STAGES = ("compare", "figures")

#: The artifact(s) each stage leaves behind, as globs relative to the stage's own
#: output directory (``<prefix>/<model>`` for a model stage, ``<prefix>`` for a
#: model-free one).  ``--resume`` requires *both* the ``ok`` status in
#: ``run_state.json`` *and* a present artifact before skipping a stage: the state file
#: can say ``ok`` for a stage whose file was lost, and a file can be present from a run
#: whose state file was overwritten.
#:
#: Each entry names the file the *next* stage actually opens, not merely one the stage
#: happens to write: `detect` is resumed on the `.npz` the ablations load (its `.json`
#: sidecar carries the conditions and is checked too), and `figures` on the manifest it
#: writes last, so a half-drawn figure set is not mistaken for a finished one.
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "describe": ("model_info.json",),
    "detect": ("scores_next_step.npz", "scores_next_step.json"),
    "mask": ("masking_curve.json",),
    "qa": ("task_qa.json",),
    "cot": ("task_cot.json",),
    "compare": ("correlation.json",),
    "figures": ("figures/manifest.json",),
    "case-study": ("case_study.json",),
}


#: Stages whose artifact can record *partial* success.  `figures` deliberately does not
#: fail when one plot breaks (a broken figure must not abort a finished run), so the skip
#: has to read its manifest rather than trust the status: otherwise a set that is missing
#: a PDF is `ok` and skipped on every later resume.
PARTIAL_ARTIFACTS: dict[str, str] = {"figures": "failed"}

#: Whether a stage *reads* other stages' artifacts.  A derived stage's own argv is only
#: paths, so its fingerprint also carries the commands of every producer in the plan: the
#: grid itself lives in `SCALES`, `scripts/` is deliberately outside `measurement_sha256`,
#: and without this an edit to `SCALES['a100']['mask']` would re-run `mask` and then skip
#: `compare`/`figures` over the tree they are supposed to describe.
#:
#: Deliberately coarse -- every producer in the plan, not a per-stage dependency graph:
#: the stages marked here are the cheap ones (seconds to a minute), so over-invalidating
#: them costs nothing, while the expensive producers keep their own exact argv.  The
#: mapping covers every stage, so adding one to `STAGES` without deciding fails a test.
STAGE_IS_DERIVED: dict[str, bool] = {
    "describe": False, "detect": False, "mask": False, "qa": False, "cot": False,
    "compare": True, "figures": True, "case-study": True,
}


class StageRecord(NamedTuple):
    """What ``--resume`` needs to know about a recorded stage."""

    argv: str | None          # the fingerprint, or None if it predates fingerprints
    code_sha256: str | None   # the *full* hash (scripts included), for the warning only


def registry_fingerprint(models: list[str]) -> str:
    """The registry entries this run actually uses, for the resume fingerprint.

    ``configs/models.json`` decides two things no stage's argv carries: which files a
    checkpoint *is* (``--verify-hashes`` checks them against it, and a mismatch aborts
    the job before any stage) and the dtype when ``--dtype`` is not passed.  Hashing only
    the entries in use means an edit to some other model does not invalidate a finished
    stage, while a change to this run's model does -- the alternative was to trust a
    registry that nothing hashes at all.

    Only the fields that decide the numbers are hashed, not the resolved absolute path:
    the same entry reached from a different layout must not re-run a stage (the job mounts
    the tree at `/job`, a local run has it elsewhere, and the digest set is what makes two
    checkpoints the same file).
    """
    from retrieval_heads.cli import resolve_model

    used = {}
    for key in models:
        _path, settings = resolve_model(key)
        used[key] = {name: settings.get(name) for name in ("dtype", "files", "shards")}
    blob = json.dumps(used, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_inputs_fingerprint(*, profile: str, models: list[str], dtype: str | None,
                           seed: int, lengths: list[int] | None, limit: int | None,
                           no_chat_template: bool, measurement_sha256: str,
                           registry_sha256: str) -> str:
    """The run-level conditions a stage's own argv does not carry.

    Fingerprinting the stage argv alone is not enough, and the gap is not hypothetical:
    ``compare``, ``figures`` and ``case-study`` take only paths, so a relaunch with a
    different ``--lengths`` re-runs ``mask`` (its argv changed) and then *skips* the
    derived stages -- a tree half from each grid, reported as a success.  ``dtype``
    changes the arithmetic without appearing in any stage's argv,
    ``measurement_sha256`` is the hash of the package that computes the numbers, and
    ``registry_sha256`` covers the effective checkpoint/dtype settings (see
    :func:`registry_fingerprint`), so an ``ok`` recorded under any of those being
    different is not trusted.

    Deliberately absent: ``--weights`` (a job's input directory is named per job, so
    including it would disable resume entirely; the checkpoints' identity is what
    ``registry_sha256`` and ``--verify-hashes`` pin) and ``--out-prefix`` (it is in every
    stage's argv, and it is where the state file itself lives).  Note the consequence of
    the latter: staging a tree under a *different* prefix -- which
    ``a100-resume.yaml`` does by design -- makes every recorded argv differ, so the first
    resume of a foreign tree re-runs its whole stage list.  That is the safe direction
    and it is cheap (the config lists only the tail), but it is not "only the failed
    stage", and the config says so.
    """
    return json.dumps({
        "profile": profile, "models": models, "dtype": dtype, "seed": seed,
        "lengths": lengths, "limit": limit, "no_chat_template": no_chat_template,
        "measurement_sha256": measurement_sha256, "registry_sha256": registry_sha256,
    }, sort_keys=True)


def stage_fingerprint(argv: list[str], run_inputs: str,
                      producers: list[list[str]] = ()) -> str:
    """The exact command a stage ran with, for ``--resume`` to compare against.

    ``run_state.json`` used to record only ``stage/model/status``, so ``--resume``
    trusted a stale ``ok``: re-launching a continuation config with a different
    ``--lengths`` (which ``a100.yaml``'s own comment invites) or with
    ``--no-chat-template`` would skip the stage and leave a tree assembled from two
    grids, reported as a success.  With the fingerprint the skip means "this exact
    command already ran here", and a mismatch is printed instead of being acted on.
    ``shlex.join`` rather than ``" ".join`` so an argument containing a space cannot
    make two different commands look alike.

    ``producers`` are the commands of the stages whose artifacts this one reads (see
    :data:`STAGE_IS_DERIVED`), so a `SCALES` edit that re-runs a producer also
    invalidates the stage that consumes its output.
    """
    parts = [run_inputs, shlex.join(argv)]
    parts.extend(shlex.join(other) for other in producers)
    return " :: ".join(parts)


def plan_argv(plan: list[tuple[str, str | None]], *, profile: str, models: list[str],
              prefix: Path, seed: int, lengths: list[int] | None = None,
              limit: int | None = None,
              no_chat_template: bool = False) -> dict[tuple[str, str | None], list[list[str]]]:
    """Build every stage's argv once, so the fingerprint and the run cannot disagree.

    Model-needing stages get their own model (the plan is model-major within a segment);
    a model-free stage gets the whole list.
    """
    return {
        (stage, model): stage_argv(
            stage, profile=profile, models=[model] if model else models, prefix=prefix,
            seed=seed, lengths=lengths, limit=limit,
            no_chat_template=no_chat_template)
        for stage, model in plan
    }


def producer_argvs(argvs_by_pair: dict[tuple[str, str | None], list[list[str]]]
                   ) -> list[list[str]]:
    """The commands of every non-derived stage in the plan, in plan order."""
    return [argv for (stage, _model), argvs in argvs_by_pair.items()
            if not STAGE_IS_DERIVED[stage] for argv in argvs]


def _artifact_is_complete(stage: str, directory: Path) -> bool:
    """Presence of every glob in :data:`STAGE_ARTIFACTS`, plus no recorded partial set."""
    if not all(any(directory.glob(pattern)) for pattern in STAGE_ARTIFACTS[stage]):
        return False
    key = PARTIAL_ARTIFACTS.get(stage)
    if key is None:
        return True
    try:
        payload = json.loads(
            (directory / STAGE_ARTIFACTS[stage][0]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        # Same principle as `run_state.json`: a file that is not the expected shape means
        # "do not skip", never "crash before the first stage".
        return False
    return not payload.get(key)


def completed_stages(prefix: Path) -> dict[tuple[str, str | None], StageRecord]:
    """``(stage, model) -> StageRecord`` for the stages that may be skipped.

    A pair is here when ``run_state.json``'s *last* status for it is ``ok`` (the file is
    an append-only log) and its artifact is complete (see :func:`_artifact_is_complete`;
    for ``figures`` that includes "the manifest lists no failure").  The record carries
    the command line that produced it -- ``None`` for a record written before the
    fingerprint existed, which the caller skips *with a warning*: the alternative would
    re-pay for the very stages this flag exists to protect.

    The first split A100 launch (`bt1u3ja8cb0it4klqehl`) is why the flag exists at all:
    the hybrid's ``mask`` ran for 75 minutes and completed, then ``case-study`` died on
    a bf16 bug, so a relaunch would have re-bought the mask -- the expensive stage --
    because a stage's granularity is per model and the plan is model-major.
    ``run_state.json`` already recorded the ``ok``; nothing read it back.

    A state file that is missing, unreadable or not shaped like ``{"stages": [...]}``
    means "skip nothing", never "skip everything".
    """
    path = prefix / "run_state.json"
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers JSONDecodeError *and* bad UTF-8
        print(f"[entry] --resume: cannot read {path}; running every stage", flush=True)
        return {}
    entries = state.get("stages") if isinstance(state, dict) else None
    if not isinstance(entries, list):
        print(f"[entry] --resume: {path} has no `stages` list; running every stage",
              flush=True)
        return {}
    last: dict[tuple[str, str | None], tuple[str, str | None, str | None]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (entry.get("stage"), entry.get("model"))
        if key[0] in STAGE_ARTIFACTS and entry.get("status"):
            last[key] = (entry["status"], entry.get("argv"), entry.get("code_sha256"))
    done: dict[tuple[str, str | None], StageRecord] = {}
    for (stage, model), (status, argv, code) in last.items():
        if status != "ok":
            continue
        directory = prefix / model if model else prefix
        if _artifact_is_complete(stage, directory):
            done[(stage, model)] = StageRecord(argv=argv, code_sha256=code)
    return done


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--weights", help="checkpoint root on the job VM (the ${WEIGHTS} input dir)")
    source.add_argument("--download-weights", action="store_true",
                        help="fetch the checkpoints from HuggingFace inside the job instead")
    parser.add_argument("--bootstrap-venv", metavar="PATH", default=None,
                        help="create/reuse a persistent venv at PATH (a ${DS_PROJECT_HOME} path) "
                             "and install requirements-datasphere.txt into it, then exit; "
                             "no checkpoints are needed for this")
    parser.add_argument("--inspect-dir", metavar="PATH", default=None, action="append",
                        help="read-only report on a directory (repeatable); writes and deletes "
                             "nothing, so it is safe to point at other users' paths")
    parser.add_argument("--project-home", metavar="PATH", default=os.environ.get("DS_PROJECT_HOME"),
                        help="root of the shared project disk, surveyed read-only before we "
                             "write anything (pass ${DS_PROJECT_HOME})")
    parser.add_argument("--use-venv", metavar="PATH", default=None,
                        help="re-exec into this existing venv before running stages (a "
                             "${DS_PROJECT_HOME} path). DataSphere rejects a cmd whose first "
                             "token is not a recognisable python, so the cached venv cannot be "
                             "the entry point -- the driver hops into it instead")
    parser.add_argument("--models", nargs="+", default=["qwen3.5-0.8b", "qwen3-0.6b"],
                        help="registry keys from configs/models.json")
    parser.add_argument("--stages", default="describe,detect,mask,qa,cot,compare,figures",
                        help=f"comma-separated subset of {','.join(STAGES)}")
    # Two different namespaces used to share the word "profile": the CLI's
    # `--profile {smoke,laptop,paper}` (which sets the detection grid) and this
    # driver's scale (which additionally knows about `t4` and `a100`).  `--scale` is
    # the honest name; `--profile` stays as an alias so the shipped configs keep
    # working.
    parser.add_argument("--scale", "--profile", dest="profile", default="laptop",
                        choices=sorted(SCALES),
                        help="job scale: per-stage flags for this machine")
    parser.add_argument("--out-prefix", default="ds-results",
                        help="results root, relative to the job working dir")
    parser.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                        help="override the registry dtype (bfloat16 on GPU)")
    parser.add_argument("--seed", type=int, default=0)
    # Preflight overrides: the scale still defines the grid, and these two flags
    # replace/limit the axes named, so "try the A100 on a short length first" is a
    # command line rather than an edit to `SCALES` that must be reverted.
    parser.add_argument("--lengths", type=int, nargs="*", default=None,
                        help="replace the --lengths values of every stage that has them "
                             "(a preflight at 1024 4096, say)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap `detect` to the first N instances in grid order "
                             "(ignored by stages that have no such flag)")
    # The sink geometry is the other axis the preflight has to price: with the chat
    # template the attention sink at position 0 sits *before* the haystack, so it can
    # never suppress criterion (2); the paper's template-free prompt puts it inside
    # `x`.  Choosing that variant used to mean editing a committed job config (exactly
    # what docs/datasphere-findings.md section 16 warns against), so it is a launch
    # flag now: it is appended to every stage whose prompts it changes.
    parser.add_argument("--no-chat-template", action="store_true",
                        help="render prompts without the chat template (the paper's "
                             "geometry); appended to detect/mask/qa/cot/case-study")
    # Presence is the cheap check `download_models.sh` already did before uploading;
    # recomputing the digests is a few seconds and catches a truncated shard at job
    # start instead of when the model fails to load.  The A100 configs pass it.
    parser.add_argument("--verify-hashes", action="store_true",
                        help="recompute the registry SHA-256 of every checkpoint file "
                             "(~3.3 GB, a few seconds) instead of only checking presence")
    # Resume is opt-in, and the monolithic configs leave it off: a *fresh* full run must
    # never skip a stage because a stale artifact happens to sit in the prefix.  The
    # continuation configs (a100-detect, a100-mask, a100-resume) pass it, which is where
    # it matters.
    parser.add_argument("--resume", action="store_true",
                        help="skip every (stage, model) that run_state.json records as ok "
                             "and whose artifact is present in --out-prefix (for "
                             "re-launching the tail of a run whose earlier stages paid "
                             "for themselves)")
    args = parser.parse_args(argv)
    if not args.bootstrap_venv and not args.inspect_dir and not (args.weights or args.download_weights):
        parser.error("one of --weights / --download-weights is required "
                     "(unless --bootstrap-venv or --inspect-dir)")
    return args


# --------------------------------------------------------------------------- venv cache
def find_stable_interpreter() -> str:
    """A Python that will still exist in the *next* job's container.

    Two requirements, both learned the hard way:

    * the platform's own venv lives at ``/job/.job_python_venv_<random>``, so a
      venv created from it is dead the moment the job ends -- we must build from
      the image's system interpreter, whose prefix is stable;
    * the version must match the running interpreter **exactly**.  The image ships
      more than one (``/usr/bin/python3`` is 3.11 while the job runs 3.10), and
      the requirements file is a lock of ``cp310`` wheels, which a 3.11 venv
      cannot install.
    """
    major, minor = sys.version_info[:2]
    base = sys.base_prefix
    candidates: list[str] = []
    if base and base != sys.prefix:
        candidates.append(str(Path(base) / "bin" / f"python{major}.{minor}"))
    candidates.append(f"/usr/bin/python{major}.{minor}")
    if base:
        candidates.append(str(Path(base) / "bin" / "python3"))
    candidates.append("/usr/bin/python3")

    probe = (
        "import sys; print(int(sys.prefix == sys.base_prefix "
        f"and sys.version_info[:2] == ({major}, {minor})))"
    )
    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        done = subprocess.run([candidate, "-c", probe], capture_output=True, text=True)
        if done.returncode == 0 and done.stdout.strip() == "1":
            return candidate
    raise SystemExit(
        f"[entry] no non-virtualenv python{major}.{minor} found in this image; "
        f"tried {candidates}"
    )


def claim_target(target: Path) -> bool:
    """Return True if ``target`` is a venv we own; refuse to touch anything else.

    The project disk is shared with other users' venvs, repositories and files, so
    the only safe rule is: create our own leaf directory, mark it, and never write
    into a non-empty directory that lacks our marker.
    """
    marker = target / ".rh-venv-owner"
    if not target.exists():
        return False
    if marker.exists():
        return True
    leftovers = list(target.iterdir())
    if not leftovers:
        return False
    raise SystemExit(
        f"[entry] refusing to touch {target}\n"
        f"[entry]   It exists, is not empty, and has no {marker.name} marker, so it may\n"
        f"[entry]   belong to another user on the shared project disk.\n"
        f"[entry]   Pass a different --bootstrap-venv path (and never delete it by hand)."
    )


def survey_project_disk(home: Path | None) -> None:
    """Read-only listing of the shared disk, so we can see whose files are there.

    The project disk is shared with other users' venvs, repositories and files.
    We never write outside our own namespaced leaf directory, but printing the
    root makes that guarantee visible in the job log instead of assumed.
    """
    if home is None:
        print("[entry] --project-home not given; skipping the shared-disk survey")
        return
    if not home.exists():
        print(f"[entry] project home {home} does not exist")
        return
    entries = sorted(home.iterdir())
    print(f"[entry] survey of {home}: {len(entries)} entries (read-only, nothing touched)")
    for entry in entries[:60]:
        kind = "dir " if entry.is_dir() else "file"
        print(f"[entry]   {kind} {entry.name}")
    if len(entries) > 60:
        print(f"[entry]   ... and {len(entries) - 60} more")


def inspect_dir(target: Path) -> None:
    """Print what is inside ``target`` -- strictly read-only.

    Exists to answer "did we touch anything that was not ours?" without relying on
    guesswork, and to let a suspicious path be examined without any risk of
    modifying it.
    """
    print(f"\n[entry] --- read-only inspection of {target}")
    if not target.exists():
        print("[entry]   does not exist")
        return
    try:
        entries = sorted(target.iterdir())
    except PermissionError as exc:
        print(f"[entry]   not readable: {exc}")
        return
    print(f"[entry]   {len(entries)} entries:")
    for entry in entries[:60]:
        kind = "dir " if entry.is_dir() else "file"
        try:
            stat = entry.stat()
            print(f"[entry]     {kind} {entry.name:40s} {stat.st_size:>12d} B  "
                  f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))}")
        except OSError as exc:
            print(f"[entry]     {kind} {entry.name:40s} (stat failed: {exc})")

    cfg = target / "pyvenv.cfg"
    if cfg.exists():
        print(f"[entry]   {cfg.name}:")
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            print(f"[entry]     {line}")

    for site in sorted(target.glob("lib/python*/site-packages")):
        packages = sorted(p.name for p in site.iterdir() if not p.name.startswith("_"))
        print(f"[entry]   {site.relative_to(target)}: {len(packages)} entries "
              f"(showing up to 25)")
        print(f"[entry]     {', '.join(packages[:25])}")


def bootstrap_venv(target: Path, requirements: Path) -> int:
    """Build the reusable venv on the project disk and verify it imports.

    Deliberately conservative about the shared disk: a unique leaf directory, an
    ownership marker, and an abort rather than an overwrite if anything unexpected
    is already there.
    """
    target = target.expanduser()
    marker = target / ".rh-venv-owner"
    ours = claim_target(target)

    base = find_stable_interpreter()
    print(f"[entry] bootstrapping venv at {target}")
    print(f"[entry]   base interpreter: {base} (this job runs {sys.version.split()[0]})")
    print(f"[entry]   running under    : {sys.executable} (this job's throwaway venv)")
    print(f"[entry]   existing venv    : {'yes, ours' if ours else 'no'}")

    if not ours:
        target.mkdir(parents=True, exist_ok=True)
        # --copies makes the interpreter independent of the base venv's symlink.
        subprocess.run([base, "-m", "venv", "--copies", str(target)], check=True)
        marker.write_text(
            "owner=ai360-heads-in-the-haystack\n"
            f"python={sys.version_info.major}.{sys.version_info.minor}\n"
            "This directory was created by scripts/datasphere_job.py --bootstrap-venv.\n"
            "Safe to delete only if you created it.\n",
            encoding="utf-8",
        )

    python = target / "bin" / "python"
    version = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"[entry]   venv python     : {version}")
    if tuple(int(x) for x in version.split(".")[:2]) != sys.version_info[:2]:
        raise SystemExit(
            f"[entry] venv is python {version} but the lock is for "
            f"{sys.version_info.major}.{sys.version_info.minor}; refusing to install "
            f"incompatible wheels"
        )

    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()[:16]
    stamp = target / ".rh-venv-stamp"
    if ours and stamp.exists() and f"requirements_sha256_16={digest}" in stamp.read_text(encoding="utf-8"):
        verify = ("import torch, transformers, matplotlib, numpy, tqdm; "
                  "print(torch.__version__, transformers.__version__)")
        done = subprocess.run([str(python), "-c", verify], capture_output=True, text=True)
        if done.returncode == 0:
            total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            print(f"[entry] venv already matches {digest} ({total / 2**30:.2f} GiB); "
                  f"skipping install")
            print(f"[entry]   {done.stdout.strip()}")
            return 0
        print("[entry]   stamp matches but the imports fail; reinstalling")

    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    # --no-deps for the same reason the job config uses it: the file is already a
    # complete exact lock, and pip 25.1.1 crashes in get_topological_weights when
    # a project is required twice (transformers and tokenizers both want
    # huggingface-hub).
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                    "-r", str(requirements)], check=True)

    # Report the optional kernels explicitly.  `causal-conv1d` is a CUDA extension
    # compiled at install time, so a failure here is the difference between the
    # hybrid running fused kernels and silently falling back to the PyTorch path --
    # and `torch.cuda.get_arch_list()` says whether the wheel covers the GPU we
    # intend to use (sm_80 for an A100).
    check = (
        "import sys, torch, transformers, matplotlib, numpy, tqdm\n"
        "print('[entry]   python', sys.version.split()[0])\n"
        "print('[entry]   torch', torch.__version__, '| cuda', torch.cuda.is_available())\n"
        "print('[entry]   transformers', transformers.__version__)\n"
        "print('[entry]   torch arch list:', torch.cuda.get_arch_list())\n"
        "for name in ('fla', 'causal_conv1d'):\n"
        "    try:\n"
        "        __import__(name); print(f'[entry]   {name}: importable')\n"
        "    except Exception as exc:\n"
        "        print(f'[entry]   {name}: NOT importable ({type(exc).__name__}: {exc})')\n"
        "try:\n"
        "    from fla.ops.gated_delta_rule import chunk_gated_delta_rule\n"
        "    print('[entry]   fla.ops.gated_delta_rule: ok')\n"
        "except Exception as exc:\n"
        "    print(f'[entry]   fla.ops.gated_delta_rule: MISSING ({type(exc).__name__}: {exc})')\n"
        # The prefill runs through SDPA, which picks the flash kernel on bf16 -- but
        # only if this GPU and this torch build support it.  Forcing the backend makes
        # the answer explicit instead of "it probably falls back to math".
        "import torch.nn.attention as attn\n"
        "try:\n"
        "    with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):\n"
        "        q = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16, device='cuda')\n"
        "        torch.nn.functional.scaled_dot_product_attention(q, q, q)\n"
        "    print('[entry]   SDPA flash backend: ok')\n"
        "except Exception as exc:\n"
        "    print(f'[entry]   SDPA flash backend: NOT available ({type(exc).__name__}: {exc})')\n"
    )
    subprocess.run([str(python), "-c", check], check=True)

    stamp.write_text(f"requirements_sha256_16={digest}\npython={version}\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"[entry] venv ready: {total / 2**30:.2f} GiB, stamp {stamp} ({digest})")
    return 0


def strip_flag(argv: list[str], flag: str) -> list[str]:
    """Drop ``--flag VALUE`` / ``--flag=VALUE`` so a re-exec does not repeat it."""
    kept, index = [], 0
    while index < len(argv):
        if argv[index] == flag:
            index += 2
            continue
        if argv[index].startswith(f"{flag}="):
            index += 1
            continue
        kept.append(argv[index])
        index += 1
    return kept


def reexec_into_venv(target: Path) -> None:
    """Replace this process with the cached venv's interpreter, if not already there.

    ``os.execv`` keeps the same PID and passes our argv through unchanged (minus
    the flags that were consumed), so the rest of ``main`` runs exactly as it
    would have.  The platform rejects a ``cmd`` whose first token is not a
    recognisable python, so this hop is how the cached venv becomes the runtime.
    """
    target = target.expanduser()
    python = target / "bin" / "python"
    if not python.exists():
        raise SystemExit(
            f"[entry] cached venv not found at {python}.\n"
            f"[entry]   build it once with the bootstrap job:\n"
            f"[entry]   cmd: python3 scripts/datasphere_job.py --project-home ${{DS_PROJECT_HOME}} "
            f"--bootstrap-venv {target}"
        )
    if Path(sys.prefix).resolve() == target.resolve():
        return
    if os.environ.get("RH_VENV_ACTIVE") == str(target):  # loop guard
        raise SystemExit(f"[entry] re-exec into {target} did not take effect")

    forwarded = strip_flag(strip_flag(sys.argv[1:], "--use-venv"), "--bootstrap-venv")
    print(f"[entry] re-exec -> {python} (from {sys.executable})", flush=True)
    os.environ["RH_VENV_ACTIVE"] = str(target)
    os.execv(str(python), [str(python), os.path.abspath(__file__), *forwarded])


def report_environment() -> None:
    """Print the accelerator situation up front.

    The GPU configurations are the least-exercised part of this setup, so the job
    log should state plainly which device the stages will actually run on rather
    than leaving it to be inferred from timings.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard requirement
        # Expected in the *cached* setup: the job starts in the platform venv,
        # which only has requirements-platform.txt, and hops into the real venv
        # right after.  Say so, rather than looking like an installation failure.
        print(f"[entry] torch not importable in this interpreter ({exc!r}).")
        print("[entry]   This is expected before the --use-venv re-exec; the real "
              "environment is reported again afterwards.")
        return
    print(f"[entry] torch {torch.__version__}")
    # The optional kernels are part of the *measurement*, not just of the speed: the
    # hybrid's 18 linear layers take a different (fused) path when
    # `flash-linear-attention` is importable, so two runs that differ here are not the
    # same experiment.  Every artifact records `provenance.optional_kernels`, but that
    # is only readable after the grid; this prints it for the interpreter that will
    # actually run the stages -- i.e. again after the `--use-venv` re-exec.  Printed
    # after the torch check on purpose: in the platform venv (pre-re-exec) everything
    # is missing, and a wall of "NOT importable" there would be noise, not a signal.
    for label, module in (("flash-linear-attention", "fla"),
                          ("causal-conv1d", "causal_conv1d")):
        try:
            found = importlib.util.find_spec(module) is not None
            print(f"[entry] optional kernel {label}: "
                  f"{'importable' if found else 'NOT importable (module not found)'}")
        except Exception as exc:  # noqa: BLE001 - report it, never fail the job here
            print(f"[entry] optional kernel {label}: NOT available "
                  f"({type(exc).__name__}: {exc})")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[entry] GPU: {props.name} sm_{props.major}{props.minor} "
              f"{props.total_memory / 2**30:.1f} GiB | cuda {torch.version.cuda} | "
              f"{torch.cuda.device_count()} device(s)")
        # The prefill runs through SDPA, which picks the flash kernel in bf16 -- but
        # only if this GPU and this torch build support it.  The chunked-prefill
        # memory bound assumes it does: on the math fallback the score matrix is
        # materialised as (heads, chunk, seq), which at 49K/8192 is ~12.9 GB for the
        # hybrid (8 Q-heads, fp32) and ~25.8 GB for the dense model (16) -- inside the
        # A100's 80 GB either way, but not inside the 22 GiB L4 that OOM'd on this.
        # The bootstrap job probed this on the L4; doing it here means every job's log
        # answers it for the card it actually got, before the grid rather than after it.
        try:
            import torch.nn.attention as attn

            with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):
                q = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16, device="cuda")
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            print("[entry] SDPA flash backend: ok")
        except Exception as exc:  # noqa: BLE001 - report it, never fail the job here
            print(f"[entry] SDPA flash backend: NOT available "
                  f"({type(exc).__name__}: {exc}) -- bf16 SDPA will use another "
                  f"backend and the chunked-prefill memory bound may not hold")
    else:
        print("[entry] GPU: none -- stages will run on CPU")


def verify_weights(root: Path, registry: Path, keys: list[str] | None = None, *,
                   hashes: bool = False) -> None:
    """Every file the registry pins must exist under ``root``.

    Presence only by default -- it is cheap, and the weights were hash-checked by
    ``download_models.sh`` before they were ever uploaded.  ``hashes=True`` also
    recomputes the SHA-256 the registry pins (``--verify-hashes``): that is a few
    seconds for ~3.3 GB, against an hour of GPU time if a truncated shard is only
    discovered when the model fails to load.  The A100 configs pass the flag; the
    cheap scales keep the presence check.

    A half-downloaded tree must not be mistaken for a ready one: the previous code
    printed "already present, leaving it alone" and carried on.
    """
    data = json.loads(registry.read_text(encoding="utf-8"))["models"]
    if keys is not None:
        unknown = [k for k in keys if k not in data]
        if unknown:
            raise SystemExit(f"[entry] --models {unknown} are not in {registry}")
        data = {k: data[k] for k in keys}
    missing: list[str] = []
    mismatched: list[str] = []
    hashed = 0
    for key, entry in data.items():
        leaf = root / Path(entry["path"]).name
        pinned = {**(entry.get("files") or {}), **(entry.get("shards") or {})}
        for name, digest in pinned.items():
            target = leaf / name
            if not target.exists() or target.stat().st_size == 0:
                missing.append(f"{key}/{name}")
                continue
            if not hashes:
                continue
            actual = _sha256(target)
            hashed += target.stat().st_size
            if digest and actual != digest:
                mismatched.append(f"{key}/{name} (registry {digest[:12]}..., file "
                                  f"{actual[:12]}...)")
    if missing:
        raise SystemExit(
            f"[entry] the checkpoint tree under {root} is incomplete: {len(missing)} "
            f"pinned file(s) missing or empty, e.g. {missing[:3]}. Refusing to start a "
            f"job on a partial download -- run scripts/download_models.sh, or fix --weights."
        )
    if mismatched:
        raise SystemExit(
            f"[entry] {len(mismatched)} pinned file(s) do not match their registry "
            f"SHA-256, e.g. {mismatched[:3]}. A corrupted checkpoint loads as garbage "
            f"weights (or not at all) hours into the job -- re-download it."
        )
    if hashes:
        print(f"[entry] checkpoints verified: every pinned file is present under {root} "
              f"and matches its SHA-256 ({hashed / 1e9:.2f} GB hashed)")
    else:
        print(f"[entry] checkpoints verified: every pinned file is present under {root} "
              f"(sizes only; pass --verify-hashes to recompute the registry digests)")


def prepare_models(args: argparse.Namespace) -> None:
    """Make the checkpoints visible as ``./models``, which the registry paths expect."""
    registry = Path(os.environ.get("RETRIEVAL_HEADS_MODELS_JSON", "configs/models.json"))
    root = Path("models").absolute()
    if root.is_symlink() and not root.exists():
        # `exists()` follows the link, so a dangling one looked "already present"
        # and verify_weights then failed instead of the link being recreated.
        print(f"[entry] removing a dangling {root} symlink -> {os.readlink(root)!r}")
        root.unlink()
    if root.is_symlink() or root.exists():
        print(f"[entry] {root} already present, leaving it alone")
        verify_weights(root, registry, args.models, hashes=args.verify_hashes)
        return
    if args.download_weights:
        print("[entry] downloading checkpoints into ./models (this takes a few minutes)")
        env = {**os.environ, "MODELS_DIR": "models"}
        subprocess.run(["bash", "scripts/download_models.sh"], check=True, env=env)
        verify_weights(root, registry, args.models, hashes=args.verify_hashes)
        return
    source = Path(args.weights).absolute()
    if not source.is_dir():
        raise SystemExit(f"[entry] --weights {source} is not a directory")
    os.symlink(source, root, target_is_directory=True)
    print(f"[entry] linked {root} -> {source}: {sorted(p.name for p in source.iterdir())}")
    verify_weights(root, registry, args.models, hashes=args.verify_hashes)


def override_dtype(models: list[str], dtype: str) -> None:
    """Point the stages at a runtime registry with the requested precision.

    The tracked ``configs/models.json`` is an *input* and is left untouched; a
    copy is written next to it and exported through
    ``RETRIEVAL_HEADS_MODELS_JSON``, which the CLI prefers.  (Rewriting the
    tracked file left a stray modification in the working tree and mutated job
    input rather than output.)
    """
    source = Path("configs/models.json")
    runtime = Path("configs/models.runtime.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    changed = []
    for key in models:
        if key not in data["models"]:
            raise SystemExit(f"[entry] {key!r} is not in {source}: {sorted(data['models'])}")
        if data["models"][key].get("dtype") != dtype:
            data["models"][key]["dtype"] = dtype
            changed.append(key)
    runtime.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.environ["RETRIEVAL_HEADS_MODELS_JSON"] = str(runtime)
    print(f"[entry] dtype={dtype} for {changed or 'nobody (already correct)'} "
          f"(runtime registry {runtime})")


def run_script(stage: str, script: Path, argv: list[str]) -> None:
    """Run a standalone script in this process, reusing the resident model.

    ``scripts/case_study.py`` (the paper's Fig. 1) imports ``_load`` from
    ``retrieval_heads.cli``, and that module keeps one model resident -- so running it
    in-process means the figure costs no second weight load, which is what made it
    worth making a stage at all.  The script ends with ``sys.exit(main())``, so exit 0
    is success, exit 1 is its own "no copy step found" signal for a figure it cannot
    draw (warn and continue: it is the last stage, and aborting there would throw away
    a finished run), and anything else fails the stage.
    """
    import runpy

    print(f"\n[entry] === {stage}: {script} {' '.join(argv)}", flush=True)
    saved = sys.argv
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            return
        if code == 1:
            print(f"[entry] stage {stage} produced no figure ({exc!r}); continuing",
                  flush=True)
            return
        print(f"[entry] stage {stage} exited with {exc!r}", flush=True)
        raise SystemExit(code) from exc
    finally:
        sys.argv = saved


def run_cli(stage: str, argv: list[str]) -> None:
    """Execute one stage **in this process**.

    Not a subprocess, because ``retrieval_heads.cli._load`` keeps one model resident
    and the driver runs a model's stages back to back: a full run used to pay the
    ~50 s weight load once per stage per model (ten times), now twice.  The
    CLI/driver contract is still the argv list from :func:`stage_argv`, so the
    existing contract tests keep their meaning.

    A failing stage raises, which aborts the job exactly as ``check=True`` did --
    and the service still collects everything already written under ``outputs``,
    which is how the detect/mask artifacts of a failed run were salvaged.
    """
    from retrieval_heads import cli

    if stage == "case-study":
        run_script(stage, Path(__file__).with_name("case_study.py"), argv)
        return

    print(f"\n[entry] === {stage}: {' '.join(argv)}", flush=True)
    try:
        rc = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print(f"[entry] stage {stage} exited with {exc!r}", flush=True)
        raise SystemExit(code) from exc
    if rc:
        raise SystemExit(f"[entry] stage {stage} returned {rc}")


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming SHA-256 of one file (the shards are >1 GB, so never read it whole)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256(*roots: Path) -> str:
    """Hash the uploaded code so a job artifact can be tied to its source.

    The job gets `local-paths` without `.git`, so `provenance()["git_rev"]` is None
    there; this is the substitute.  Covers `retrieval_heads/` **and** `scripts/` --
    `SCALES` in this driver defines the grid, so a run must not hash as unchanged
    when the grid changed.  Hashes path + contents of every `.py` in sorted order.
    """
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def record_stage_state(prefix: Path, stage: str, model: str | None, status: str,
                       error: str | None = None, argv: str | None = None,
                       code_sha256: str | None = None) -> None:
    """Append one line of provenance to ``<prefix>/run_state.json``.

    Written *before* a stage starts (``running``) and again when it ends, so a job
    that dies half-way still tells whoever picks up the pieces which stages
    finished.  The outputs of a failed job are collected by the service, which is
    what makes a resume possible at all (see ``configs/datasphere/t4-resume.yaml``).

    ``argv`` is the fingerprint of the command the stage ran with (see
    :func:`stage_fingerprint`); ``--resume`` compares it against the command it is about
    to run, so a relaunch with a changed grid re-runs the stage instead of silently
    keeping an artifact from another measurement.  ``code_sha256`` is the *full* hash and
    is only reported when it moves -- the part that must invalidate a stage
    (``retrieval_heads/``) is already inside the fingerprint.
    """
    path = prefix / "run_state.json"
    state: dict[str, Any] = {"stages": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            # A file that is not the expected shape would otherwise crash the *first*
            # record of the next run, i.e. before any stage has a chance to run.
            if isinstance(loaded, dict) and isinstance(loaded.get("stages"), list):
                state = loaded
        except (OSError, ValueError):
            # ValueError covers JSONDecodeError and a truncated multi-byte character:
            # a job killed mid-write leaves exactly that, and it must not take the next
            # run down before its first stage.
            pass
    entry: dict[str, Any] = {
        "stage": stage, "model": model, "status": status,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if argv is not None:
        entry["argv"] = argv
    if code_sha256 is not None:
        entry["code_sha256"] = code_sha256
    if error:
        entry["error"] = error[:300]
    state.setdefault("stages", []).append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def stage_plan(stages: list[str], models: list[str]) -> list[tuple[str, str | None]]:
    """Order ``(stage, model)`` pairs so that each model is loaded once per segment.

    The config's stage order *is* the run order.  Within every run of consecutive
    model-needing stages the models are grouped (model-major), so one checkpoint is
    loaded once and the next load evicts it; a model-free stage (``compare``,
    ``figures``) ends the current segment, and anything listed after it runs after the
    model-free stages have.

    That last part is not cosmetic.  The A100 configs list ``case-study`` *after*
    ``compare,figures`` on purpose: the figure stage is the one that runs a standalone
    script, and it is the one that crashed on bf16 (job `bt1u3ja8cb0it4klqehl`, three
    seconds in, after 75 minutes of masking).  A cheap unproven stage must not be able
    to take an expensive one down with it -- and with ``--resume`` it must not block the
    stage behind it on every relaunch either.  The price is two extra weight loads
    (tens of seconds on the A100), paid once.
    """
    plan: list[tuple[str, str | None]] = []
    for kind, group in itertools.groupby(stages, key=lambda stage: stage in MODEL_STAGES):
        stages_here = list(group)
        if kind:
            for model in models:
                plan.extend((stage, model) for stage in stages_here)
        else:
            plan.extend((stage, None) for stage in stages_here)
    return plan


def apply_grid_overrides(argv: list[str], *, lengths: list[int] | None = None,
                         limit: int | None = None) -> list[str]:
    """Replace a stage's `--lengths` values and/or append `--limit`.

    The scales in `SCALES` are what a *job* runs, so until now the advice in
    `configs/datasphere/a100.yaml` ("start with a short length before the full grid")
    could not be followed without editing the scale table -- and forgetting to revert
    that edit.  These two flags make a preflight a command-line decision: the grid
    still comes from the scale, and only the axes named here are overridden.
    """
    if lengths is not None:
        values = [str(value) for value in lengths]
        if "--lengths" in argv:
            start = argv.index("--lengths")
            end = start + 1
            while end < len(argv) and not argv[end].startswith("--"):
                end += 1
            argv = argv[:start] + ["--lengths", *values] + argv[end:]
        else:
            argv = [*argv, "--lengths", *values]
    if limit is not None and "--limit" not in argv:
        argv = [*argv, "--limit", str(limit)]
    return argv


def stage_argv(
    stage: str,
    *,
    profile: str,
    models: list[str],
    prefix: Path,
    seed: int,
    lengths: list[int] | None = None,
    limit: int | None = None,
    no_chat_template: bool = False,
) -> list[list[str]]:
    """Build the ``retrieval_heads.cli`` argument lists for one stage.

    Kept as a pure function so the CLI/driver contract can be tested without a
    job: a mismatch here otherwise costs a DataSphere round trip (tens of seconds
    with the cached venv, minutes when the environment is built) before it
    surfaces, and that already happened once with ``--seed``.

    ``lengths``/``limit``/``no_chat_template`` are the preflight overrides (see
    :func:`apply_grid_overrides`); ``limit`` is silently ignored by stages that have
    no such flag, and ``no_chat_template`` is appended only to the stages that render
    prompts (describe/compare/figures do not).
    """
    runs = {key: prefix / key for key in models}
    scales = SCALES[profile]
    # Only the prompt-building stages take it; `add_common` defines the flag on all of
    # them, but passing it to a stage that never renders a prompt would be noise.
    prompt_flags = ["--no-chat-template"] if no_chat_template else []

    if stage == "describe":
        return [["describe", "--model", key, "--out", str(runs[key])] for key in models]
    if stage == "detect":
        flags = apply_grid_overrides(list(scales["detect"]), lengths=lengths, limit=limit)
        return [
            ["detect", "--model", key, "--out", str(runs[key]),
             "--seed", str(seed), *flags, *prompt_flags]
            for key in models
        ]
    if stage in ("mask", "qa", "cot"):
        # `--limit` is a `detect` flag only (it caps the instance grid), and `--lengths`
        # exists on `detect`/`mask` only -- `qa`/`cot` build their items from `--data`
        # or the built-ins and have no grid at all, so appending the override to them
        # made argparse exit 2 and failed the stage.  Both are gated here rather than
        # in `apply_grid_overrides`, which cannot know the stage.
        flags = apply_grid_overrides(list(scales[stage]),
                                     lengths=lengths if stage in LENGTH_STAGES else None)
        return [
            [stage, "--model", key, "--out", str(runs[key]),
             "--seed", str(seed), *flags, *prompt_flags]
            for key in models
        ]
    if stage == "compare":
        if len(runs) < 2:
            return []
        return [["compare", "--runs", *[str(p) for p in runs.values()], "--out", str(prefix)]]
    if stage == "case-study":
        # The paper's Fig. 1 needs the real attention rows and therefore a model, which
        # the driver already has resident at this point; `--scores` reuses the run's own
        # recorded conditions, and the per-model `--out` keeps two models' JSON apart.
        # `--length 4096` (not the script's 1024 default) so the figure shows a needle
        # further from the query than the trivial case, and `--max-new-tokens 96` so it
        # generates under the same budget as the run it illustrates; the length cannot
        # match a 1K-49K grid, and `store_rows` keeps every step's row, so a longer one
        # would be the figure's whole cost.
        return [["--model", key, "--scores", str(runs[key]), "--out", str(runs[key]),
                 "--length", "4096", "--max-new-tokens", "96", "--seed", str(seed),
                 *prompt_flags]
                for key in models]
    if stage == "figures":
        return [["figures", "--runs", *[str(p) for p in runs.values()],
                 "--out", str(prefix / "figures")]]
    raise SystemExit(f"[entry] unknown stage {stage!r}; known: {list(STAGES)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"[entry] unknown stages {unknown}; known: {list(STAGES)}")

    print(f"[entry] cwd={os.getcwd()} python={sys.version.split()[0]}")
    # Stages now run in this process, so keep progress visible: a pipe would
    # otherwise block-buffer it until the stage ends.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass
    if args.inspect_dir:
        report_environment()
        for path in args.inspect_dir:
            inspect_dir(Path(path).expanduser())
        if not args.bootstrap_venv and not (args.weights or args.download_weights):
            return 0
    if args.bootstrap_venv:
        report_environment()
        survey_project_disk(Path(args.project_home).expanduser() if args.project_home else None)
        rc = bootstrap_venv(Path(args.bootstrap_venv).expanduser(),
                            Path("scripts/requirements-datasphere.txt"))
        if rc != 0 or not args.use_venv:
            return rc
        # Fall through: re-exec into the venv we just built and run the stages,
        # so one job both creates the cache and proves it works.
    if args.use_venv:
        reexec_into_venv(Path(args.use_venv))
        # os.execv replaces the process; reaching here means it did not happen.
    # Report again in whatever interpreter we ended up in: in the cached setup the
    # only place the GPU is actually visible is *after* the re-exec.
    report_environment()
    prepare_models(args)
    if args.dtype:
        override_dtype(args.models, args.dtype)

    prefix = Path(args.out_prefix)
    # Tie the artifacts to the exact code that produced them: the job has no `.git`,
    # so `git_rev` is None and this hash is the only link back to a revision.
    code = code_sha256(Path("retrieval_heads"), Path("scripts"))
    os.environ["RH_CODE_SHA256"] = code
    print(f"[entry] code_sha256={code}", flush=True)
    if args.lengths is not None and not args.lengths:
        raise SystemExit("--lengths was given but empty; pass at least one length")
    if args.lengths is not None or args.limit is not None or args.no_chat_template:
        print(f"[entry] grid override: lengths={args.lengths} limit={args.limit} "
              f"no_chat_template={args.no_chat_template} "
              f"(the scale's own values are replaced, not extended)", flush=True)
    # The package's own hash is part of every stage's fingerprint: an `ok` recorded by
    # another revision of the code that computes the numbers must not be reused.  The
    # full hash (scripts included) is recorded next to it and only *reported* when it
    # moves -- a change to `case_study.py` or to this driver cannot invalidate a masking
    # curve, and re-running an hour of `mask` for it would defeat the flag.
    measurement = code_sha256(Path("retrieval_heads"))
    registry = registry_fingerprint(args.models)
    run_inputs = run_inputs_fingerprint(
        profile=args.profile, models=args.models, dtype=args.dtype, seed=args.seed,
        lengths=args.lengths, limit=args.limit,
        no_chat_template=args.no_chat_template, measurement_sha256=measurement,
        registry_sha256=registry)
    done = completed_stages(prefix) if args.resume else {}
    if args.resume:
        skipped = sorted((stage, model or "*") for stage, model in done)
        print(f"[entry] --resume: {len(skipped)} finished stage(s) will be skipped: "
              f"{skipped or 'none'}", flush=True)
    plan = stage_plan(stages, args.models)
    argvs_by_pair = plan_argv(plan, profile=args.profile, models=args.models,
                              prefix=prefix, seed=args.seed, lengths=args.lengths,
                              limit=args.limit,
                              no_chat_template=args.no_chat_template)
    producers = producer_argvs(argvs_by_pair)
    for stage, model in plan:
        argvs = argvs_by_pair[(stage, model)]
        # A derived stage reads other stages' artifacts, so their commands are part of
        # what makes its own result valid (see STAGE_IS_DERIVED).
        extra = producers if STAGE_IS_DERIVED[stage] else []
        label = f"{stage} ({model or 'all models'})"
        record = done.get((stage, model))
        if record is not None:
            now = [stage_fingerprint(argv, run_inputs, extra) for argv in argvs]
            if not argvs:
                # `compare` with a single model builds no command at all (it needs two
                # runs), so there is nothing to skip or run -- say that instead of
                # printing "running it" and then doing nothing.
                print(f"[entry] --resume: {label} has no command to run (it needs two "
                      f"models); nothing to do", flush=True)
            elif record.argv is None:
                print(f"[entry] --resume: skipping {label} -- ok + artifact present; its "
                      f"record predates the argv fingerprint, so the command cannot be "
                      f"compared", flush=True)
                continue
            elif record.argv in now:
                if record.code_sha256 and record.code_sha256 != code:
                    print(f"[entry] --resume: {label} was recorded by code "
                          f"{record.code_sha256}, now running {code}; skipping it anyway "
                          f"-- the artifact records its own provenance, so a mixed tree "
                          f"is detectable, and re-running on every edit is not a safe "
                          f"default either", flush=True)
                print(f"[entry] --resume: skipping {label} -- ok + artifact present, same "
                      f"command line", flush=True)
                continue
            else:
                # Not silent: a stale `ok` from another grid is exactly the tree this
                # flag must not assemble, so say which command produced the artifact.
                print(f"[entry] --resume: {label} is recorded ok but for a different "
                      f"command; running it.\n[entry]   recorded: {record.argv}\n"
                      f"[entry]   now     : {now[0] if now else '(no command)'}",
                      flush=True)
        for cli_argv in argvs:
            fingerprint = stage_fingerprint(cli_argv, run_inputs, extra)
            record_stage_state(prefix, stage, model, "running", argv=fingerprint,
                               code_sha256=code)
            try:
                run_cli(stage, cli_argv)
            except BaseException as exc:  # noqa: BLE001 - record, then abort
                record_stage_state(prefix, stage, model, "failed", str(exc),
                                   argv=fingerprint, code_sha256=code)
                raise
            record_stage_state(prefix, stage, model, "ok", argv=fingerprint,
                               code_sha256=code)

    print("\n[entry] artifacts under", prefix.absolute())
    for path in sorted(prefix.rglob("*")):
        if path.is_file():
            print(f"[entry]   {path} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
