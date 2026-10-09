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
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
    #: and 8 heads a 8192-token chunk peaks around 6.4 GiB for that matrix, so the
    #: protection stays while the number of chunks halves.  One-shot (`0`) was the
    #: first version of this profile and it was a bad trade: it saves ~4-8% of the
    #: attention time and gives up the bound entirely.
    "a100": {
        "detect": ["--profile", "paper", "--argmax-domain", "haystack",
                   "--prefill-chunk", "8192", "--max-new-tokens", "96",
                   # Build all 270 prompts (CPU only) before the first forward pass:
                   # one prompt whose haystack cannot be located verbatim would
                   # otherwise abort the stage hours into the run.  The `paper`/`t4`
                   # scales leave it off -- there a mid-grid failure costs minutes.
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
#: Stages that need a loaded model (and therefore benefit from the one-model cache).
MODEL_STAGES = ("describe", "detect", "mask", "qa", "cot", "case-study")
#: Stages that read every model's artifacts and need no model at all.
MODEL_FREE_STAGES = ("compare", "figures")


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
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[entry] GPU: {props.name} sm_{props.major}{props.minor} "
              f"{props.total_memory / 2**30:.1f} GiB | cuda {torch.version.cuda} | "
              f"{torch.cuda.device_count()} device(s)")
    else:
        print("[entry] GPU: none -- stages will run on CPU")


def verify_weights(root: Path, registry: Path, keys: list[str] | None = None) -> None:
    """Every file the registry pins must exist under ``root``.

    Cheap on purpose -- it checks presence, not sha256 (hashing 3.2 GB on every job
    start would cost GPU time).  A half-downloaded tree must not be mistaken for a
    ready one: the previous code printed "already present, leaving it alone" and
    carried on.
    """
    data = json.loads(registry.read_text(encoding="utf-8"))["models"]
    if keys is not None:
        unknown = [k for k in keys if k not in data]
        if unknown:
            raise SystemExit(f"[entry] --models {unknown} are not in {registry}")
        data = {k: data[k] for k in keys}
    missing: list[str] = []
    for key, entry in data.items():
        leaf = root / Path(entry["path"]).name
        pinned = {**(entry.get("files") or {}), **(entry.get("shards") or {})}
        for name in pinned:
            target = leaf / name
            if not target.exists() or target.stat().st_size == 0:
                missing.append(f"{key}/{name}")
    if missing:
        raise SystemExit(
            f"[entry] the checkpoint tree under {root} is incomplete: {len(missing)} "
            f"pinned file(s) missing or empty, e.g. {missing[:3]}. Refusing to start a "
            f"job on a partial download -- run scripts/download_models.sh, or fix --weights."
        )
    print(f"[entry] checkpoints verified: every pinned file is present under {root}")


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
        verify_weights(root, registry, args.models)
        return
    if args.download_weights:
        print("[entry] downloading checkpoints into ./models (this takes a few minutes)")
        env = {**os.environ, "MODELS_DIR": "models"}
        subprocess.run(["bash", "scripts/download_models.sh"], check=True, env=env)
        verify_weights(root, registry, args.models)
        return
    source = Path(args.weights).absolute()
    if not source.is_dir():
        raise SystemExit(f"[entry] --weights {source} is not a directory")
    os.symlink(source, root, target_is_directory=True)
    print(f"[entry] linked {root} -> {source}: {sorted(p.name for p in source.iterdir())}")
    verify_weights(root, registry, args.models)


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
                       error: str | None = None) -> None:
    """Append one line of provenance to ``<prefix>/run_state.json``.

    Written *before* a stage starts (``running``) and again when it ends, so a job
    that dies half-way still tells whoever picks up the pieces which stages
    finished.  The outputs of a failed job are collected by the service, which is
    what makes a resume possible at all (see ``configs/datasphere/t4-resume.yaml``).
    """
    path = prefix / "run_state.json"
    state: dict[str, Any] = {"stages": []}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"stages": []}
    entry: dict[str, Any] = {
        "stage": stage, "model": model, "status": status,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if error:
        entry["error"] = error[:300]
    state.setdefault("stages", []).append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def stage_plan(stages: list[str], models: list[str]) -> list[tuple[str, str | None]]:
    """Order ``(stage, model)`` pairs so that each model is loaded once.

    Model-needing stages are grouped per model (model-major): everything for model A
    runs while A is resident, then the next load evicts it.  The model-free stages
    (``compare``, ``figures``) read every model's artifacts, so they come last.
    """
    plan: list[tuple[str, str | None]] = []
    for model in models:
        for stage in stages:
            if stage in MODEL_STAGES:
                plan.append((stage, model))
    for stage in stages:
        if stage in MODEL_FREE_STAGES:
            plan.append((stage, None))
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
        # `--limit` is a `detect` flag only (it caps the instance grid); passing it to
        # a stage without one would make argparse exit.
        flags = apply_grid_overrides(list(scales[stage]), lengths=lengths)
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
        return [["--model", key, "--scores", str(runs[key]), "--out", str(runs[key]),
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
    os.environ["RH_CODE_SHA256"] = code_sha256(Path("retrieval_heads"), Path("scripts"))
    print(f"[entry] code_sha256={os.environ['RH_CODE_SHA256']}", flush=True)
    if args.lengths is not None and not args.lengths:
        raise SystemExit("--lengths was given but empty; pass at least one length")
    if args.lengths is not None or args.limit is not None or args.no_chat_template:
        print(f"[entry] grid override: lengths={args.lengths} limit={args.limit} "
              f"no_chat_template={args.no_chat_template} "
              f"(the scale's own values are replaced, not extended)", flush=True)
    for stage, model in stage_plan(stages, args.models):
        targets = [model] if model else args.models
        for cli_argv in stage_argv(stage, profile=args.profile, models=targets,
                                   prefix=prefix, seed=args.seed,
                                   lengths=args.lengths, limit=args.limit,
                                   no_chat_template=args.no_chat_template):
            record_stage_state(prefix, stage, model, "running")
            try:
                run_cli(stage, cli_argv)
            except BaseException as exc:  # noqa: BLE001 - record, then abort
                record_stage_state(prefix, stage, model, "failed", str(exc))
                raise
            record_stage_state(prefix, stage, model, "ok")

    print("\n[entry] artifacts under", prefix.absolute())
    for path in sorted(prefix.rglob("*")):
        if path.is_file():
            print(f"[entry]   {path} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
