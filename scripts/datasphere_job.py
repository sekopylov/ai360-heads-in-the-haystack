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

#: Per-stage flags for each scale, kept in step with the two reproduce scripts.
#:
#: Head counts are given as *fractions* (``--k-frac``) rather than absolute K,
#: because K=8 is 17% of Qwen3.5-0.8B's 48 scoreable heads but only 1.8% of
#: Qwen3-0.6B's 448 -- absolute K silently makes the two models incomparable.
SCALES: dict[str, dict[str, list[str]]] = {
    "smoke": {
        "detect": ["--profile", "smoke"],
        "mask": ["--k-frac", "0.04", "0.17", "--lengths", "1024", "--random-trials", "1"],
        "qa": ["--k-frac", "0.08", "--random-trials", "1"],
        "cot": ["--k-frac", "0.08", "--random-trials", "1", "--max-new-tokens", "128"],
    },
    "laptop": {
        "detect": ["--profile", "laptop"],
        "mask": ["--k-frac", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "1024", "--random-trials", "2"],
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "2"],
        "cot": ["--k-frac", "0.08", "--random-trials", "1", "--max-new-tokens", "192"],
    },
    #: Single T4 (gt4.1 / gt4i.1, 16-24 GB VRAM).  Much larger than `laptop`,
    #: deliberately smaller than `paper`: the mask stage re-runs the prefill for
    #: every configuration, so its cost grows with contexts x K values x trials.
    "t4": {
        "detect": ["--profile", "paper",
                   "--lengths", "1024", "2048", "4096", "8192", "16384",
                   "--depths", "5", "--needles", "3"],
        "mask": ["--k-frac", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "4096", "8192", "--random-trials", "3"],
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "3"],
        "cot": ["--k-frac", "0.08", "--random-trials", "2", "--max-new-tokens", "256"],
    },
    "paper": {
        # `--profile paper` is the 3x7x10 grid; reproduce_gpu.sh widens it to the
        # paper's exact 20 lengths, so the job does the same.
        "detect": ["--profile", "paper",
                   "--lengths", "1024", "2048", "4096", "8192", "16384",
                   "24576", "32768", "40960", "49152"],
        "mask": ["--k-frac", "0.01", "0.02", "0.04", "0.08", "0.17", "0.33",
                 "--lengths", "4096", "8192", "16384", "--random-trials", "5"],
        "qa": ["--k-frac", "0.04", "0.08", "0.17", "--random-trials", "5"],
        "cot": ["--k-frac", "0.08", "--random-trials", "3"],
    },
}

STAGES = ("describe", "detect", "mask", "qa", "cot", "compare", "figures")


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
    parser.add_argument("--profile", default="laptop", choices=sorted(SCALES))
    parser.add_argument("--out-prefix", default="ds-results",
                        help="results root, relative to the job working dir")
    parser.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                        help="override the registry dtype (bfloat16 on GPU)")
    parser.add_argument("--seed", type=int, default=0)
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

    check = (
        "import sys, torch, transformers, matplotlib, numpy, tqdm; "
        "print('[entry]   python', sys.version.split()[0]); "
        "print('[entry]   torch', torch.__version__, '| cuda', torch.cuda.is_available()); "
        "print('[entry]   transformers', transformers.__version__)"
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


def prepare_models(args: argparse.Namespace) -> None:
    """Make the checkpoints visible as ``./models``, which the registry paths expect."""
    root = Path("models").absolute()
    if root.is_symlink() or root.exists():
        print(f"[entry] {root} already present, leaving it alone")
        return
    if args.download_weights:
        print("[entry] downloading checkpoints into ./models (this takes a few minutes)")
        env = {**os.environ, "MODELS_DIR": "models"}
        subprocess.run(["bash", "scripts/download_models.sh"], check=True, env=env)
        return
    source = Path(args.weights).absolute()
    if not source.is_dir():
        raise SystemExit(f"[entry] --weights {source} is not a directory")
    os.symlink(source, root, target_is_directory=True)
    print(f"[entry] linked {root} -> {source}: {sorted(p.name for p in source.iterdir())}")


def override_dtype(models: list[str], dtype: str) -> None:
    """Rewrite the registry so the models load in the requested precision."""
    registry = Path("configs/models.json")
    data = json.loads(registry.read_text(encoding="utf-8"))
    changed = []
    for key in models:
        if key not in data["models"]:
            raise SystemExit(f"[entry] {key!r} is not in {registry}: {sorted(data['models'])}")
        if data["models"][key].get("dtype") != dtype:
            data["models"][key]["dtype"] = dtype
            changed.append(key)
    registry.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[entry] dtype={dtype} for {changed or 'nobody (already correct)'}")


def run(stage: str, argv: list[str]) -> None:
    # sys.executable is the cached venv's python when --use-venv re-exec'd us.
    cmd = [sys.executable, "-m", "retrieval_heads.cli", *argv]
    print(f"\n[entry] === {stage}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def stage_argv(
    stage: str,
    *,
    profile: str,
    models: list[str],
    prefix: Path,
    seed: int,
) -> list[list[str]]:
    """Build the ``retrieval_heads.cli`` argument lists for one stage.

    Kept as a pure function so the CLI/driver contract can be tested without a
    job: a mismatch here costs a full ~9-minute DataSphere round trip (env build)
    before it surfaces, and that already happened once with ``--seed``.
    """
    runs = {key: prefix / key for key in models}
    scales = SCALES[profile]

    if stage == "describe":
        return [["describe", "--model", key, "--out", str(runs[key])] for key in models]
    if stage == "detect":
        return [
            ["detect", "--model", key, "--out", str(runs[key]),
             "--seed", str(seed), *scales["detect"]]
            for key in models
        ]
    if stage in ("mask", "qa", "cot"):
        return [
            [stage, "--model", key, "--out", str(runs[key]),
             "--seed", str(seed), *scales[stage]]
            for key in models
        ]
    if stage == "compare":
        if len(runs) < 2:
            return []
        return [["compare", "--runs", *[str(p) for p in runs.values()], "--out", str(prefix)]]
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
    for stage in stages:
        for cli_argv in stage_argv(stage, profile=args.profile, models=args.models,
                                   prefix=prefix, seed=args.seed):
            run(stage, cli_argv)

    print("\n[entry] artifacts under", prefix.absolute())
    for path in sorted(prefix.rglob("*")):
        if path.is_file():
            print(f"[entry]   {path} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
