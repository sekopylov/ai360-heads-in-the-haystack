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


def test_stage_plan_is_model_major_so_each_model_loads_once():
    """Order the stages so the one-model cache can actually be reused.

    A full run used to load each checkpoint once per stage (ten loads, ~50 s each
    on the job GPU); grouping a model's stages keeps it to two.
    """
    driver = load_job_driver()
    models = ["qwen3.5-0.8b", "qwen3-0.6b"]
    plan = driver.stage_plan(["describe", "detect", "mask", "qa", "cot",
                              "compare", "figures"], models)

    assert set(driver.MODEL_STAGES) | set(driver.MODEL_FREE_STAGES) == set(driver.STAGES)
    assert not set(driver.MODEL_STAGES) & set(driver.MODEL_FREE_STAGES)

    # Every model stage of the first model comes before any stage of the second.
    first_second_model = min(i for i, (_, model) in enumerate(plan)
                             if model == models[1])
    assert all(model != models[1] for _, model in plan[:first_second_model])
    assert {model for _, model in plan[:first_second_model]} == {models[0]}

    # Model-free stages need every model's artifacts, so they run last.
    assert [stage for stage, model in plan if model is None] == ["compare", "figures"]
    assert all(model is not None for stage, model in plan
               if stage in driver.MODEL_STAGES)


def test_run_state_records_every_stage(tmp_path):
    """A job that dies half-way must still say how far it got."""
    import json

    driver = load_job_driver()
    driver.record_stage_state(tmp_path, "detect", "qwen3-0.6b", "running")
    driver.record_stage_state(tmp_path, "detect", "qwen3-0.6b", "ok")
    driver.record_stage_state(tmp_path, "mask", "qwen3-0.6b", "failed", "boom")

    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert [(e["stage"], e["status"]) for e in state["stages"]] == [
        ("detect", "running"), ("detect", "ok"), ("mask", "failed")]
    assert state["stages"][-1]["error"] == "boom"


def test_load_keeps_exactly_one_model_resident(monkeypatch):
    """`_load` reuses the resident model and evicts the previous one."""
    import retrieval_heads.cli as cli
    import retrieval_heads.models as models

    from tests.test_regressions import attention_info

    calls: list[str] = []

    def fake_load(path, *, dtype, attn_implementation, device):
        calls.append(path)
        return object(), object(), attention_info(1, 2)

    monkeypatch.setattr(models, "load_model", fake_load)
    monkeypatch.setattr(models, "describe_model", lambda info: "census")
    monkeypatch.setattr(cli, "resolve_model",
                        lambda name: (f"/models/{name}", {"dtype": "float32"}))
    monkeypatch.setattr(cli, "_LOADED", {})
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    cli._load("a")
    cli._load("a")
    assert calls == ["/models/a"], "the second call reloaded the same model"

    cli._load("b")
    assert calls == ["/models/a", "/models/b"]
    assert len(cli._LOADED) == 1, "two models stayed resident"


def test_code_sha256_ties_artifacts_to_the_uploaded_code(tmp_path):
    """A job has no `.git`, so `git_rev` is None -- this hash is the substitute."""
    driver = load_job_driver()
    package = tmp_path / "retrieval_heads"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n", encoding="utf-8")
    (package / "b.py").write_text("y = 2\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\x00")

    first = driver.code_sha256(package)
    assert first == driver.code_sha256(package), "not deterministic"
    # Byte-compiled files must not move the hash (they are regenerated, not source).
    (package / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\x01\x02")
    assert driver.code_sha256(package) == first

    (package / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert driver.code_sha256(package) != first


def test_provenance_records_the_code_hash(monkeypatch):
    from retrieval_heads.provenance import provenance

    monkeypatch.delenv("RH_CODE_SHA256", raising=False)
    assert "code_sha256" not in provenance()
    monkeypatch.setenv("RH_CODE_SHA256", "deadbeefdeadbeef")
    assert provenance()["code_sha256"] == "deadbeefdeadbeef"


def test_mask_takes_capture_method_from_the_scores_not_the_parser():
    """`mask` has no `--capture-method`; defaulting to "patch" mislabelled the run."""
    from types import SimpleNamespace

    from retrieval_heads.cli import resolve_detection_settings

    # The flags `mask` actually has; it has no --capture-method by design.
    mask_args = SimpleNamespace(system_prompt=None, no_chat_template=False, thinking=False)
    scores = SimpleNamespace(meta={"config": {"capture_method": "output_attentions",
                                             "threshold": 0.1}})
    settings = resolve_detection_settings(mask_args, scores)
    assert settings.capture_method == "output_attentions"

    # Nothing recorded -> the historical default, not an exception.
    settings = resolve_detection_settings(mask_args, SimpleNamespace(meta={}))
    assert settings.capture_method == "patch"


def test_code_sha256_covers_scripts_too(tmp_path):
    """`SCALES` lives in scripts/, so a grid change must move the hash."""
    driver = load_job_driver()
    package = tmp_path / "retrieval_heads"
    scripts = tmp_path / "scripts"
    package.mkdir()
    scripts.mkdir()
    (package / "a.py").write_text("x = 1\n", encoding="utf-8")
    (scripts / "job.py").write_text("SCALES = {}\n", encoding="utf-8")

    before = driver.code_sha256(package, scripts)
    (scripts / "job.py").write_text("SCALES = {'t4': 1}\n", encoding="utf-8")
    assert driver.code_sha256(package, scripts) != before


def test_ablation_arguments_reject_silent_zero_runs():
    """`--random-trials 0` / `--max-new-tokens 0` used to produce NaN artifacts.

    `detect` already refused a zero budget; the ablations accepted it and reported
    all-zero metrics, and `np.mean([])` left `null` in the JSON.  The helper is also
    why `--mixer-trials` must be read with `getattr`: only `mask` has it.
    """
    from types import SimpleNamespace

    from retrieval_heads.cli import require_ablation_args

    require_ablation_args(SimpleNamespace(max_new_tokens=8, random_trials=3, mixer_trials=2))

    for bad, message in ((SimpleNamespace(max_new_tokens=0, random_trials=3), "max-new-tokens"),
                         (SimpleNamespace(max_new_tokens=8, random_trials=0), "random-trials"),
                         (SimpleNamespace(max_new_tokens=8, random_trials=3,
                                          mixer_trials=0), "mixer-trials")):
        with pytest.raises(SystemExit, match=message):
            require_ablation_args(bad)

    # `qa`/`cot` have no --mixer-trials; the helper must not require it.
    require_ablation_args(SimpleNamespace(max_new_tokens=8, random_trials=1))


def test_provenance_records_which_optional_kernels_were_installed():
    """The hybrid takes a fused path when these are present, so it is provenance.

    `transformers` imports them lazily and silently falls back to PyTorch, so two runs
    of the same code are not necessarily the same experiment.
    """
    from retrieval_heads.provenance import provenance

    kernels = provenance()["optional_kernels"]
    assert set(kernels) == {"flash_linear_attention", "causal_conv1d"}
    assert all(isinstance(v, bool) for v in kernels.values())


def test_every_command_only_reads_flags_its_parser_defines():
    """No `args.<flag>` a command reaches may be absent from its subparser.

    The first version of this test only scanned the `cmd_*` bodies, so it missed
    `resolve_detection_settings` reading `args.corpus` -- which `qa`/`cot` do not
    define, and the GPU job died on the `qa` stage with an AttributeError.  The
    check now walks the intra-module call graph from each command, so shared
    helpers are covered too, while `getattr(args, ..., default)` stays allowed.
    """
    import ast

    from retrieval_heads import cli

    source = (REPO_ROOT / "retrieval_heads" / "cli.py").read_text(encoding="utf-8")
    reads: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        reads[node.name] = {
            child.attr for child in ast.walk(node)
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
            and child.value.id == "args"
        }
        calls[node.name] = {
            child.func.id for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }

    def reachable_flags(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen if seen is not None else set()
        if name in seen:
            return set()
        seen.add(name)
        flags = set(reads.get(name, ()))
        for callee in calls.get(name, ()):
            flags |= reachable_flags(callee, seen)
        return flags

    parser = cli.build_parser()
    for command, function in (("detect", cli.cmd_detect), ("mask", cli.cmd_mask),
                              ("qa", cli.cmd_qa), ("cot", cli.cmd_cot)):
        namespace = vars(parser.parse_args([command, "--model", "m"]))
        missing = sorted(reachable_flags(function.__name__) - set(namespace))
        assert not missing, f"{command}: {function.__name__} reaches undefined flags {missing}"
