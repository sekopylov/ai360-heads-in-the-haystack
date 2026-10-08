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
               "paper.yaml"}


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
