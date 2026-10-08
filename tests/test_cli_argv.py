"""The job driver's CLI contract.

``scripts/datasphere_job.py`` builds ``retrieval_heads.cli`` argument vectors and
runs them inside a DataSphere job.  A mismatch between the two (a flag the CLI does
not accept, or a global option placed after the subcommand) is only discovered
after a full job round trip -- tens of seconds with the cached venv, minutes when
the platform builds the environment -- before the entry point even starts.  That
already happened once, so the contract is pinned here.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from retrieval_heads.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_job_driver():
    spec = importlib.util.spec_from_file_location(
        "datasphere_job", REPO_ROOT / "scripts" / "datasphere_job.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["datasphere_job"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return load_job_driver()


def test_every_stage_and_profile_builds_parseable_argv(driver):
    """Every argv the driver would run must parse cleanly."""
    parser = build_parser()
    models = ["qwen3.5-0.8b", "qwen3-0.6b"]
    prefix = Path("ds-results")
    checked = 0
    for profile in sorted(driver.SCALES):
        for stage in driver.STAGES:
            for argv in driver.stage_argv(stage, profile=profile, models=models,
                                          prefix=prefix, seed=7):
                parsed = parser.parse_args(argv)
                assert parsed.command == stage
                # `describe` carries no seed; compare/figures do not run sampling.
                if stage in ("detect", "mask", "qa", "cot"):
                    assert parsed.seed == 7, f"{profile}/{stage} lost --seed"
                checked += 1
    assert checked > 0


def test_seed_is_accepted_before_and_after_the_subcommand(driver):
    parser = build_parser()
    after = parser.parse_args(["detect", "--model", "m", "--seed", "3", "--profile", "smoke"])
    before = parser.parse_args(["--seed", "3", "detect", "--model", "m", "--profile", "smoke"])
    assert after.seed == before.seed == 3


def test_compare_and_figures_need_no_model_flag(driver):
    parser = build_parser()
    for argv in driver.stage_argv("compare", profile="t4", models=["a", "b"],
                                  prefix=Path("out"), seed=0):
        assert parser.parse_args(argv).command == "compare"
    for argv in driver.stage_argv("figures", profile="t4", models=["a", "b"],
                                  prefix=Path("out"), seed=0):
        assert parser.parse_args(argv).command == "figures"


def test_compare_is_skipped_with_a_single_model(driver):
    assert driver.stage_argv("compare", profile="t4", models=["only"],
                             prefix=Path("out"), seed=0) == []


def test_scales_use_fraction_not_absolute_k(driver):
    """Absolute K is not comparable across a 48-head and a 448-head model."""
    for profile, scales in driver.SCALES.items():
        for stage in ("mask", "qa", "cot"):
            flags = scales[stage]
            assert "--k-frac" in flags, f"{profile}/{stage} must use --k-frac"
            assert "--k" not in flags, f"{profile}/{stage} must not pin absolute K"


def test_shell_scripts_use_fraction_not_absolute_k():
    """The reproduce scripts must follow the same rule as the driver's SCALES."""
    scripts = sorted((REPO_ROOT / "scripts").glob("reproduce_*.sh"))
    assert scripts, "no reproduce scripts found"
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\w-])--k(?![\w-])", text), (
            f"{path.name}: absolute --k is not comparable across models; use --k-frac"
        )
        assert "--k-frac" in text, f"{path.name}: expected --k-frac"


def test_strip_flag_removes_both_spellings(driver):
    """The re-exec must not forward --use-venv again, or it would loop forever."""
    assert driver.strip_flag(["--use-venv", "/p", "--weights", "/w"], "--use-venv") == \
        ["--weights", "/w"]
    assert driver.strip_flag(["--use-venv=/p", "--weights", "/w"], "--use-venv") == \
        ["--weights", "/w"]
    assert driver.strip_flag(["--weights", "/w"], "--use-venv") == ["--weights", "/w"]


def test_bootstrap_needs_no_weights_but_a_normal_run_does(driver, capsys):
    ok = driver.parse_args(["--bootstrap-venv", "/disk/rh-venv"])
    assert ok.bootstrap_venv == "/disk/rh-venv"

    cached = driver.parse_args(["--use-venv", "/disk/rh-venv", "--weights", "/w"])
    assert cached.use_venv == "/disk/rh-venv"

    with pytest.raises(SystemExit) as excinfo:
        driver.parse_args(["--models", "qwen3-0.6b"])   # neither weights nor bootstrap
    assert excinfo.value.code == 2, "argparse usage errors exit with code 2"
    assert "--weights" in capsys.readouterr().err


def test_driver_has_exactly_one_main_and_it_runs_the_stages():
    """Guard against a botched edit leaving a truncated duplicate main().

    An earlier refactor left two ``def main`` definitions; the truncated one was
    silently shadowed by the complete one, so everything *looked* fine while half
    the function was dead.  Check the structure of the source directly.
    """
    import ast

    source = (REPO_ROOT / "scripts" / "datasphere_job.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    mains = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "main"]
    assert len(mains) == 1, f"expected exactly one main(), found {len(mains)}"

    called = {node.func.id for node in ast.walk(mains[0])
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "stage_argv" in called, "main() must build argv through stage_argv"
    assert "reexec_into_venv" in called, "main() must honour --use-venv"
    assert "prepare_models" in called, "main() must link the checkpoints"

    # The old assertion compared a set's size to the list's size, which is a
    # tautology; the thing worth guarding is duplicate function names.
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"duplicate top-level function names: {duplicates}"
