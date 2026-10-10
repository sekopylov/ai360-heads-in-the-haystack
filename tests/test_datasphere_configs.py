"""Static validation of the DataSphere job configs.

These files are only exercised by a real job, where a typo costs a job round trip
-- about 40 s with the cached venv, several minutes when the platform has to build
the environment -- before it surfaces.  Everything checkable offline is checked
here instead, including the actual ``cmd`` string through the driver's own parser.
"""

from __future__ import annotations

import json

import packaging.requirements
import pytest
import yaml

from tests.conftest import REPO_ROOT
from tests.test_cli_argv import load_job_driver

CONFIG_DIR = REPO_ROOT / "configs" / "datasphere"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_are_present():
    assert CONFIGS, f"no job configs in {CONFIG_DIR}"


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    if not CONFIGS:
        pytest.skip("no DataSphere configs")
    return {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in CONFIGS}


def test_every_requirements_file_parses(configs):
    """The CLI runs packaging.Requirement over every line, comments included."""
    for name, config in configs.items():
        path = config.get("env", {}).get("python", {}).get("requirements-file")
        if not path:
            continue
        req_file = REPO_ROOT / path
        assert req_file.exists(), f"{name}: {path} does not exist"
        for lineno, line in enumerate(req_file.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                packaging.requirements.Requirement(stripped)
            except Exception as exc:  # noqa: BLE001 - report the offending line
                pytest.fail(f"{name}: {path}:{lineno} is not a valid requirement "
                            f"({exc}): {stripped!r}")


def test_cmd_starts_with_python_and_targets_the_driver(configs):
    """The CLI infers the entry point from cmd; a bare filename cannot work."""
    for name, config in configs.items():
        cmd = config["cmd"]
        assert cmd.split()[0] in {"python3", "python"}, f"{name}: cmd must start with python"
        assert "scripts/datasphere_job.py" in cmd, (
            f"{name}: local-paths unpack the driver to /job/scripts/, so cmd must say "
            f"scripts/datasphere_job.py"
        )


def test_config_cmd_parses_with_the_driver(configs):
    """Run the real ``cmd`` string through the driver's own argument parser.

    Checking the first token and the driver path is not enough: a typo in
    ``--profile``, an unknown ``--stages`` entry or a model key missing from the
    registry would all pass the static checks and surface only after a job has
    started.  ``${VAR}`` placeholders are stubbed with harmless paths.
    """
    driver = load_job_driver()
    registry = json.loads(
        (REPO_ROOT / "configs" / "models.json").read_text(encoding="utf-8")
    )["models"]

    for name, config in configs.items():
        tokens = [
            token.replace("${WEIGHTS}", "/w").replace("${DS_PROJECT_HOME}", "/disk")
            for token in config["cmd"].split()
        ]
        assert tokens[0] in {"python3", "python"}, f"{name}: cmd must start with python"
        assert tokens[1] == "scripts/datasphere_job.py", (
            f"{name}: second token must be the driver path, got {tokens[1]!r}"
        )

        args = driver.parse_args(tokens[2:])
        assert args.profile in driver.SCALES, (
            f"{name}: --profile {args.profile!r} is not in SCALES {sorted(driver.SCALES)}"
        )
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown_stages = [s for s in stages if s not in driver.STAGES]
        assert not unknown_stages, f"{name}: unknown stages {unknown_stages}"
        unknown_models = [m for m in args.models if m not in registry]
        assert not unknown_models, (
            f"{name}: models not in configs/models.json: {unknown_models}"
        )


def test_project_home_requires_the_disk_flag(configs):
    for name, config in configs.items():
        if "${DS_PROJECT_HOME}" not in config["cmd"]:
            continue
        assert "attach-project-disk" in config.get("flags", []), (
            f"{name}: the CLI rejects ${{DS_PROJECT_HOME}} without attach-project-disk"
        )


def test_cached_configs_use_a_namespaced_venv_path(configs):
    """The project disk is shared; a bare ${DS_PROJECT_HOME}/rh-venv risks a collision."""
    for name, config in configs.items():
        cmd = config["cmd"]
        if "--use-venv" not in cmd and "--bootstrap-venv" not in cmd:
            continue
        assert "ai360-heads-in-the-haystack" in cmd, (
            f"{name}: the venv path must be namespaced under our own directory"
        )


def test_cached_and_bootstrap_agree_on_the_venv_path(configs):
    if "t4-cached.yaml" not in configs or "t4-bootstrap.yaml" not in configs:
        pytest.skip("cached configs not present")

    def venv_path(cmd: str) -> str:
        parts = cmd.split()
        for flag in ("--use-venv", "--bootstrap-venv"):
            if flag in parts:
                return parts[parts.index(flag) + 1]
        raise AssertionError(f"no venv flag in {cmd!r}")

    cached = venv_path(configs["t4-cached.yaml"]["cmd"])
    bootstrap = venv_path(configs["t4-bootstrap.yaml"]["cmd"])
    assert cached == bootstrap, "the cached run would not find the bootstrapped venv"


def test_outputs_are_declared_where_results_are_written(configs):
    """A stage writing outside the declared outputs silently loses its artifacts."""
    for name, config in configs.items():
        if "--inspect-dir" in config["cmd"]:
            continue  # read-only audit job: it produces no artifacts by design
        if "--bootstrap-venv" in config["cmd"] and "--use-venv" not in config["cmd"]:
            # Environment-maintenance job: it installs into the project-disk venv and
            # returns, so there is nothing to collect.
            continue
        outputs = config.get("outputs") or []
        assert outputs, f"{name}: no outputs declared"
        cmd = config["cmd"]
        for prefix in outputs:
            assert str(prefix) in cmd, (
                f"{name}: output {prefix!r} is declared but --out-prefix does not use it"
            )


#: Configs that must run on a GPU.  Listed explicitly: the previous condition
#: tested the file *name* for "gt4", so `t4.yaml` was skipped by its own check.
GPU_CONFIGS = {"t4.yaml", "t4-smoke.yaml", "t4-cached.yaml", "t4-bootstrap.yaml",
               "t4-venv.yaml", "paper.yaml", "a100.yaml"}


def test_gpu_configs_request_a_gpu_shape(configs):
    present = GPU_CONFIGS & set(configs)
    assert present == GPU_CONFIGS, f"missing GPU configs: {GPU_CONFIGS - set(configs)}"
    for name in sorted(present):
        config = configs[name]
        instances = config.get("cloud-instance-types") or config.get("cloud_instance_types") or []
        assert instances, f"{name}: no cloud-instance-types"
        assert all(str(i).startswith("g") for i in instances), (
            f"{name}: expected GPU instance types (gt4*/g2*/g1*), got {instances}"
        )


def test_requirements_lock_is_complete_for_torch(configs):
    """The lock must pin torch, transformers and their CUDA deps, or --no-deps bites."""
    locked = (REPO_ROOT / "scripts" / "requirements-datasphere.txt").read_text(encoding="utf-8")
    for needed in ("torch==", "transformers==", "numpy==", "nvidia-cublas-cu12=="):
        assert needed in locked, f"lock file is missing {needed}"


def test_the_disk_audit_job_is_strictly_read_only(configs):
    """The audit must not be able to run stages, download weights or write outputs."""
    audits = {n: c for n, c in configs.items() if "--inspect-dir" in c["cmd"]}
    if not audits:
        pytest.skip("no audit config")
    for name, config in audits.items():
        cmd = config["cmd"]
        assert "--weights" not in cmd and "--download-weights" not in cmd
        assert "--stages" not in cmd
        assert not (config.get("outputs") or []), f"{name} declares outputs"
        assert "--bootstrap-venv" not in cmd, f"{name} could create a venv"


def test_weight_verification_can_recompute_the_registry_hashes(tmp_path):
    """`--verify-hashes` is the A100's cheap insurance against a corrupt shard.

    Presence-only checking is deliberate for the cheap scales -- `download_models.sh`
    hash-checks the tree before it is ever uploaded -- but a job that costs 542.88
    RUB/h should not discover a truncated shard when the model fails to load an hour
    in.  The digests are in `configs/models.json`; recomputing them is a few seconds.
    """
    import hashlib

    driver = load_job_driver()
    leaf = tmp_path / "Qwen3-Toy"
    leaf.mkdir()
    payload = b"weights" * 1000
    shard = leaf / "model.safetensors"
    shard.write_bytes(payload)
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": {"toy": {
        "path": "models/Qwen3-Toy",
        "shards": {"model.safetensors": hashlib.sha256(payload).hexdigest()},
    }}}), encoding="utf-8")

    # The honest tree passes both modes.
    driver.verify_weights(tmp_path, registry, ["toy"])
    driver.verify_weights(tmp_path, registry, ["toy"], hashes=True)

    # One flipped byte keeps the size and the presence check happy; only the digest
    # notices.  A checkpoint this far gone loads as garbage or not at all.
    shard.write_bytes(payload[:-1] + b"z")
    driver.verify_weights(tmp_path, registry, ["toy"])
    with pytest.raises(SystemExit, match="SHA-256"):
        driver.verify_weights(tmp_path, registry, ["toy"], hashes=True)

    # A missing file is still an error in both modes.
    shard.unlink()
    for hashes in (False, True):
        with pytest.raises(SystemExit, match="incomplete"):
            driver.verify_weights(tmp_path, registry, ["toy"], hashes=hashes)


def test_a100_configs_recheck_the_checkpoint_hashes(configs):
    """The flag is wired where it pays for itself, and nowhere else by accident."""
    driver = load_job_driver()
    a100 = {name: config for name, config in configs.items() if name.startswith("a100")}
    assert a100, "no A100 configs found"
    for name, config in a100.items():
        tokens = [token.replace("${WEIGHTS}", "/w").replace("${DS_PROJECT_HOME}", "/disk")
                  for token in config["cmd"].split()]
        args = driver.parse_args(tokens[2:])
        assert args.verify_hashes, (
            f"{name}: an A100 job must recompute the registry SHA-256 before the grid"
        )

    for name in ("t4-cached.yaml", "paper.yaml"):
        if name not in configs:
            continue
        tokens = [token.replace("${WEIGHTS}", "/w").replace("${DS_PROJECT_HOME}", "/disk")
                  for token in configs[name]["cmd"].split()]
        assert not driver.parse_args(tokens[2:]).verify_hashes, (
            f"{name}: the cheap scales keep the presence-only check"
        )


def test_a100_profile_keeps_a_memory_bound_and_the_paper_grid():
    """The A100 profile trades chunk *count* for safety, not safety for speed.

    Chunking does not repeat layer work (each token belongs to one chunk); it bounds
    the attention score matrix at O(chunk x seq).  The first version of this profile
    used `--prefill-chunk 0` on the theory that 49K in 4096-token chunks cost twelve
    prefills -- it costs ~8% more attention work, and gives up the bound that keeps a
    float32 SDPA fallback from materialising (heads, seq, seq).  The grid must stay
    `paper`'s, or the A100 numbers would not be comparable to the L4's.
    """
    from pathlib import Path

    driver = load_job_driver()
    assert "a100" in driver.SCALES, sorted(driver.SCALES)

    for stage in ("detect", "mask", "qa", "cot"):
        argv = driver.stage_argv(stage, profile="a100", models=["m"],
                                 prefix=Path("ds"), seed=0)[0]
        assert "--prefill-chunk" in argv, (stage, argv)
        chunk = argv[argv.index("--prefill-chunk") + 1]
        assert chunk == "8192", (stage, chunk)
        assert chunk != "0", "one-shot gives up the memory bound for ~8% of attention"

    detect = driver.SCALES["a100"]["detect"]
    assert detect[detect.index("--max-new-tokens") + 1] == "96", detect
    # The domain is pinned, not inherited from the code default: a job's artifacts
    # must stay explainable if that default ever moves.
    assert detect[detect.index("--argmax-domain") + 1] == "haystack", detect
    # The expensive run builds all 270 prompts before the first forward pass, so one
    # prompt whose haystack cannot be located verbatim fails in minutes rather than
    # hours.  The L4 scales leave it off (there a mid-grid failure costs minutes).
    assert "--preflight" in detect, detect
    for profile in ("paper", "t4", "laptop"):
        assert "--preflight" not in driver.SCALES[profile]["detect"], profile

    # The ablation sample set is what `retrieval_std` is measured over, and its size
    # is lengths x depths x needles -- an earlier assertion multiplied only the last
    # two, so the "same 15-sample budget" comment was wrong by 3x (the a100 mask runs
    # at three lengths).
    mask = driver.SCALES["a100"]["mask"]
    assert mask[mask.index("--depths") + 1] == "5", mask
    assert mask[mask.index("--needles") + 1] == "3", mask

    def flag_list(argv: list[str], flag: str) -> list[str]:
        """Values of a `nargs="*"` flag: everything up to the next `--flag`."""
        out: list[str] = []
        for token in argv[argv.index(flag) + 1:]:
            if token.startswith("--"):
                break
            out.append(token)
        return out

    a100_lengths = flag_list(mask, "--lengths")
    per_point = (len(a100_lengths) * int(mask[mask.index("--depths") + 1])
                 * int(mask[mask.index("--needles") + 1]))
    assert a100_lengths == ["4096", "8192", "16384"], a100_lengths
    assert per_point == 45, (per_point, mask)
    # ... and it is 4.5x the t4 run's set, which is what the cost comment must say.
    # The t4 scale sets neither `--depths` nor `--needles`, so the CLI defaults (5 and
    # one held-out needle) apply: 2 x 5 x 1 = 10.
    t4_mask = driver.SCALES["t4"]["mask"]
    assert "--depths" not in t4_mask and "--needles" not in t4_mask, t4_mask
    t4_per_point = len(flag_list(t4_mask, "--lengths")) * 5
    assert t4_per_point == 10, (t4_per_point, t4_mask)

    def lengths(profile: str) -> list[str]:
        return flag_list(driver.SCALES[profile]["detect"], "--lengths")

    assert lengths("a100") == lengths("paper"), "the A100 grid drifted from `paper`"


def test_driver_can_override_the_grid_for_a_preflight():
    """The a100 config's own advice ("try a short length first") must be executable.

    The scale table is what a job runs, so without a passthrough the preflight meant
    editing `SCALES` and remembering to revert it -- and the revert is exactly the
    kind of edit that silently ships.
    """
    from pathlib import Path

    driver = load_job_driver()
    argv = driver.stage_argv("detect", profile="a100", models=["m"], prefix=Path("ds"),
                             seed=0, lengths=[1024, 4096], limit=60)[0]
    start = argv.index("--lengths")
    assert argv[start + 1:start + 3] == ["1024", "4096"], argv
    assert "49152" not in argv, "the scale's lengths were extended, not replaced"
    assert argv[argv.index("--limit") + 1] == "60"

    # `--limit` is a `detect` flag only: appending it to another stage would make
    # argparse exit, so the override must not do that.
    mask = driver.stage_argv("mask", profile="a100", models=["m"], prefix=Path("ds"),
                             seed=0, lengths=[1024], limit=60)[0]
    assert "--limit" not in mask, mask
    assert mask[mask.index("--lengths") + 1] == "1024", mask

    # With no override the scale is untouched.
    plain = driver.stage_argv("detect", profile="a100", models=["m"], prefix=Path("ds"),
                              seed=0)[0]
    assert plain == ["detect", "--model", "m", "--out", "ds/m", "--seed", "0",
                     *driver.SCALES["a100"]["detect"]], plain
