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


def test_the_committed_dense_run_stays_in_a_paper_like_band():
    """A regression tripwire on the scoring, using the committed artifact.

    The integration test that asserts a sparsity band runs against a real checkpoint
    and is not part of CI; this one reads the committed `ds-results/` and would catch
    a scoring change that moves the dense model's share by an order of magnitude.

    The band is deliberately a *guard* band, not the paper's 3-6%: the measured value
    is 6.2% (marginally above the paper's band, and the README says so).  A collapse
    to ~30% (the old bug) or to ~0% would fail here.
    """
    import json

    summary = json.loads(
        (REPO_ROOT / "ds-results" / "qwen3-0.6b" / "summary_next_step.json").read_text(
            encoding="utf-8"))
    share = summary["sparsity"]["thresholds"]["0.1"]["frac"]
    assert 0.02 <= share <= 0.12, share
    assert summary["n_instances"] == 75

    hybrid = json.loads(
        (REPO_ROOT / "ds-results" / "qwen3.5-0.8b" / "summary_next_step.json").read_text(
            encoding="utf-8"))
    # The hybrid is the documented counter-example: most of its 6 attention layers'
    # heads clear the threshold, which the README discusses rather than hides.
    assert hybrid["sparsity"]["thresholds"]["0.1"]["frac"] > 0.5


def test_every_stage_and_profile_builds_parseable_argv(driver):
    """Every argv the driver would run must parse cleanly.

    `case-study` is the one stage that is a standalone script rather than a
    `retrieval_heads.cli` subcommand (it needs the real attention rows), so it is
    checked by its own test below instead of against this parser.

    The preflight overrides are part of the contract, not a separate path: they are
    built here as well, because appending `--lengths` to a stage whose CLI has no such
    flag (`qa`/`cot`) makes argparse exit 2 and fails the stage -- and the plain build
    never noticed, since no shipped config combines them.
    """
    parser = build_parser()
    models = ["qwen3.5-0.8b", "qwen3-0.6b"]
    prefix = Path("ds-results")
    checked = 0
    for profile in sorted(driver.SCALES):
        for stage in driver.STAGES:
            if stage == "case-study":
                continue
            for overrides in ({}, {"lengths": [1024], "limit": 60}):
                for argv in driver.stage_argv(stage, profile=profile, models=models,
                                              prefix=prefix, seed=7, **overrides):
                    parsed = parser.parse_args(argv)
                    assert parsed.command == stage
                    # `describe` carries no seed; compare/figures do not run sampling.
                    if stage in ("detect", "mask", "qa", "cot"):
                        assert parsed.seed == 7, f"{profile}/{stage} lost --seed"
                    # The override may only reach a stage that defines the flag.
                    if overrides and stage in driver.LENGTH_STAGES:
                        assert parsed.lengths == [1024], (profile, stage)
                    elif overrides:
                        assert "--lengths" not in argv, (profile, stage, argv)
                    checked += 1
    assert checked > 0


def test_case_study_stage_targets_the_script_and_the_runs_own_conditions(driver):
    """Fig. 1 is a stage now: it must run the script against the run it illustrates.

    `--scores <run dir>` is what makes the script reuse the recorded conditions, and
    `run_cli` must dispatch the stage to the script rather than to `cli.main` (whose
    parser has no `case-study` subcommand).
    """
    calls: list[tuple] = []
    original = driver.run_script
    driver.run_script = lambda stage, script, argv: calls.append((stage, script, argv))
    try:
        driver.run_cli("case-study", ["--model", "m", "--scores", "ds/m", "--out", "ds/m"])
    finally:
        driver.run_script = original
    assert len(calls) == 1
    stage, script, argv = calls[0]
    assert stage == "case-study" and script.name == "case_study.py"
    assert argv == ["--model", "m", "--scores", "ds/m", "--out", "ds/m"]

    for argv in driver.stage_argv("case-study", profile="a100", models=["m", "n"],
                                  prefix=Path("ds"), seed=0):
        assert argv[argv.index("--model") + 1] in {"m", "n"}
        model = argv[argv.index("--model") + 1]
        assert argv[argv.index("--scores") + 1] == f"ds/{model}", argv
        assert argv[argv.index("--out") + 1] == f"ds/{model}", argv


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


#: The flags `SCALES` owns.  Everything else in a reproduce script (`--model`,
#: `--out`, `--dtype`, `--seed`) is not part of the grid.
GRID_FLAGS = ("--profile", "--argmax-domain", "--lengths", "--depths", "--needles",
              "--random-trials", "--max-new-tokens", "--k-frac", "--prefill-chunk")

#: Which driver scale each script is supposed to mirror.
SCRIPT_SCALES = {"reproduce_laptop.sh": "laptop", "reproduce_gpu.sh": "paper"}


def _flag_values(tokens: list[str]) -> dict[str, list[str]]:
    """``{flag: [values]}`` for a token list, ignoring positional tokens."""
    out: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            values: list[str] = []
            index += 1
            while index < len(tokens) and not tokens[index].startswith("--"):
                values.append(tokens[index])
                index += 1
            out[token] = values
        else:
            index += 1
    return out


def _script_invocations(text: str) -> dict[str, list[str]]:
    """``{stage: argv}`` for the ``-m retrieval_heads.cli <stage>`` calls in a script.

    Backslash continuations are joined and shell variables (``"${DTYPE[@]}"``) are
    dropped, so the result is the literal flag vector the script passes.
    """
    import shlex

    joined = text.replace("\\\n", " ")
    out: dict[str, list[str]] = {}
    for line in joined.splitlines():
        match = re.search(r"-m\s+retrieval_heads\.cli\s+(\w+)(.*)", line)
        if not match:
            continue
        stage, rest = match.group(1), match.group(2)
        tokens = [t for t in shlex.split(rest) if not t.startswith("${")]
        out.setdefault(stage, []).extend(tokens)
    return out


@pytest.mark.parametrize("name", sorted(SCRIPT_SCALES))
def test_reproduce_scripts_match_the_driver_grid(driver, name):
    """A full parity check against `SCALES`, not just "uses --k-frac".

    The scripts and the job driver are two implementations of the same grid, and
    they had already drifted (`reproduce_gpu.sh` left `cot` at the 192-token CLI
    default while `SCALES['paper']` pins 256), which changes what the stage measures.
    """
    path = REPO_ROOT / "scripts" / name
    assert path.exists(), f"{name} is missing"
    scale = SCRIPT_SCALES[name]
    invocations = _script_invocations(path.read_text(encoding="utf-8"))
    assert invocations, f"{name}: no retrieval_heads.cli invocations found"
    for stage in ("detect", "mask", "qa", "cot"):
        assert stage in invocations, f"{name}: no `{stage}` invocation"
        script_flags = _flag_values(invocations[stage])
        driver_flags = _flag_values(driver.SCALES[scale][stage])
        for flag in GRID_FLAGS:
            assert script_flags.get(flag) == driver_flags.get(flag), (
                f"{name}/{stage}: {flag} is {script_flags.get(flag)} in the script but "
                f"{driver_flags.get(flag)} in SCALES[{scale!r}]"
            )


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


def test_stage_plan_is_model_major_so_each_model_loads_once_per_segment():
    """Order the stages so the one-model cache can actually be reused.

    A full run used to load each checkpoint once per stage (ten loads, ~50 s each on the
    job GPU); grouping a model's stages keeps it to one load per model *per segment*.
    The count is not two for every config: a stage listed after a model-free one starts a
    new segment, so `a100.yaml`'s `...,compare,figures,case-study` costs four loads -- the
    price of not letting a figure crash abort the run between two `mask` stages (see the
    next test).  Two extra loads of tens of seconds against an hour of masking.
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

    # One segment (no model stage after `compare,figures`): exactly two loads.
    assert len(plan) == 2 * 5 + 2, plan

    # The A100 list adds `case-study` after the model-free stages: a second segment, so
    # four loads -- and both masks still finish before anything else.
    a100 = driver.stage_plan(["describe", "detect", "mask", "compare", "figures",
                              "case-study"], models)
    assert [stage for stage, _ in a100 if stage in ("mask", "case-study")] == [
        "mask", "mask", "case-study", "case-study"], a100
    assert [model for stage, model in a100 if stage == "case-study"] == models


def test_a_stage_listed_after_the_model_free_ones_runs_after_them():
    """`case-study` last means a crash in it cannot kill the expensive stages.

    The first split A100 launch is the evidence: `case-study` died three seconds in on
    a bf16 bug, and because the plan was model-major it died *between* the hybrid's
    75-minute `mask` and the dense model's -- so the dense mask never ran, and a plain
    relaunch would have hit the same stage again before reaching it.  Listing
    `case-study` after `compare,figures` puts it after every mask; the price is two
    extra weight loads, which the A100 pays in tens of seconds.
    """
    driver = load_job_driver()
    models = ["qwen3.5-0.8b", "qwen3-0.6b"]
    plan = driver.stage_plan(["mask", "compare", "figures", "case-study"], models)

    # Both models' masks finish before the first figure stage is asked for anything.
    stages = [stage for stage, _ in plan]
    assert stages == ["mask", "mask", "compare", "figures", "case-study", "case-study"], plan
    assert [model for stage, model in plan if stage == "mask"] == models
    # The segmented plan must not lose or duplicate a stage of the config.
    assert sorted(stages) == sorted(["mask", "compare", "figures", "case-study"] * 1
                                    + ["mask", "case-study"])


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

    # The argv is what `--resume` compares against, so it has to survive the round trip.
    driver.record_stage_state(tmp_path, "mask", "qwen3-0.6b", "ok",
                              argv="mask --model qwen3-0.6b")
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert state["stages"][-1]["argv"] == "mask --model qwen3-0.6b"
    assert "argv" not in state["stages"][0], "a record without an argv gained a null one"

    # A state file that is not `{"stages": [...]}` must not crash the next run's first
    # record (which would happen before any stage had a chance to start).
    (tmp_path / "run_state.json").write_text("[]", encoding="utf-8")
    driver.record_stage_state(tmp_path, "detect", "qwen3-0.6b", "ok")
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert [e["stage"] for e in state["stages"]] == ["detect"]


def test_load_keeps_exactly_one_model_resident(monkeypatch):
    """`_load` reuses the resident model and evicts the previous one."""
    import retrieval_heads.cli as cli
    import retrieval_heads.models as models

    from tests._helpers import attention_info

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


def test_run_script_executes_in_process_and_tolerates_no_figure(tmp_path):
    """The case-study stage runs a script, not a CLI subcommand.

    Four properties matter: it really runs the file with the argv it was handed; the
    script's `sys.exit(main())` means a clean 0 is success (the driver marks a stage
    failed on *any* SystemExit); its "no copy step found" signal (exit 1) only warns,
    because it is the last stage and aborting there would discard a finished run; and
    any other non-zero exit still fails the stage.
    """
    driver = load_job_driver()
    script = tmp_path / "s.py"
    out = tmp_path / "out.txt"
    script.write_text(
        "import sys, pathlib\n"
        "pathlib.Path(sys.argv[1]).write_text(' '.join(sys.argv[2:]))\n",
        encoding="utf-8",
    )
    driver.run_script("case-study", script, [str(out), "a", "b"])
    assert out.read_text(encoding="utf-8") == "a b"
    assert sys.argv[0] != str(script), "sys.argv was left pointing at the script"

    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    driver.run_script("case-study", script, [])          # success

    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    driver.run_script("case-study", script, [])          # warns, does not raise

    script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        driver.run_script("case-study", script, [])
    assert exc.value.code == 3


def test_no_chat_template_is_appended_to_the_prompt_stages_only(driver):
    """The sink geometry is a launch decision, not a committed-config edit.

    With a chat template position 0 sits before the haystack, so the sink can never win
    criterion (2); the paper's template-free prompt puts it inside `x`.  The two are
    different measurements, so the variant has to be launchable without editing
    `a100.yaml` -- and the flag must reach only the stages that render a prompt.
    """
    parser = build_parser()
    for stage in ("detect", "mask", "qa", "cot"):
        for argv in driver.stage_argv(stage, profile="a100", models=["m"],
                                      prefix=Path("ds"), seed=0, no_chat_template=True):
            assert "--no-chat-template" in argv, stage
            assert parser.parse_args(argv).no_chat_template is True, stage

    # `case-study` builds a prompt too, but it is a script rather than a CLI
    # subcommand, so it is checked by presence only.
    for argv in driver.stage_argv("case-study", profile="a100", models=["m"],
                                  prefix=Path("ds"), seed=0, no_chat_template=True):
        assert "--no-chat-template" in argv, argv

    for stage in ("describe", "compare", "figures"):
        for argv in driver.stage_argv(stage, profile="a100", models=["m", "n"],
                                      prefix=Path("ds"), seed=0, no_chat_template=True):
            assert "--no-chat-template" not in argv, (stage, argv)

    # Default: the flag is absent, so a run cannot change geometry by accident.
    for argv in driver.stage_argv("detect", profile="a100", models=["m"],
                                  prefix=Path("ds"), seed=0):
        assert "--no-chat-template" not in argv, argv


def test_the_driver_can_import_the_package_from_a_foreign_cwd(tmp_path):
    """The in-process stage runner must make `retrieval_heads` importable itself.

    Running a script puts the *script's* directory on `sys.path[0]` -- in a job that
    is `/job/scripts`, while the package `local-paths` uploads sits at `/job/retrieval_heads`.
    The old per-stage `python -m retrieval_heads.cli` subprocess got the cwd on the path
    for free (`-m` adds it); `run_cli` imports in-process instead, so the driver has to
    add its own repo root.  The first A100 preflight died on exactly this, at the first
    stage, after the venv/hash/GPU checks had all passed.
    """
    import os
    import subprocess

    driver_path = REPO_ROOT / "scripts" / "datasphere_job.py"
    snippet = (
        "import runpy\n"
        f"runpy.run_path({str(driver_path)!r}, run_name='driver_under_test')\n"
        "import retrieval_heads.cli\n"
        "print('importable')\n"
    )
    # cwd outside the repo and an empty PYTHONPATH: the only way the import can work is
    # the driver inserting its own parent, which is what a job relies on.
    done = subprocess.run([sys.executable, "-c", snippet], cwd=tmp_path,
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stderr
    assert "importable" in done.stdout


def _write_run_state(prefix: Path, entries: list[dict]) -> Path:
    import json

    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "run_state.json").write_text(json.dumps({"stages": entries}),
                                           encoding="utf-8")
    return prefix


def test_resume_skips_only_a_stage_that_is_ok_and_left_its_artifact(driver, tmp_path):
    """The split A100 launch is why this exists: 75 minutes of `mask`, lost to a crash.

    Job `bt1u3ja8cb0it4klqehl` completed the hybrid's masking curve and then died three
    seconds into `case-study`; re-running the config re-paid for the mask, because a
    stage's granularity is per model and the plan is model-major.  `run_state.json`
    already recorded the `ok` -- nothing read it back.

    Both halves of the condition matter: a stage can be `ok` with its artifact lost (a
    collection that missed it), and an artifact can be present without an `ok` (the
    state file was overwritten, or the stage died after writing).
    """
    prefix = tmp_path / "ds"
    for model in ("qwen3.5-0.8b", "qwen3-0.6b"):
        (prefix / model).mkdir(parents=True)
    (prefix / "figures").mkdir()
    # ok *and* artifact -> resumable
    for name in ("model_info.json", "masking_curve.json", "case_study.json"):
        (prefix / "qwen3.5-0.8b" / name).write_text("{}", encoding="utf-8")
    (prefix / "figures" / "manifest.json").write_text("{}", encoding="utf-8")
    # artifact present but the status is not `ok` -> still runs
    (prefix / "qwen3-0.6b" / "scores_next_step.npz").write_bytes(b"npz")
    (prefix / "qwen3-0.6b" / "scores_next_step.json").write_text("{}", encoding="utf-8")
    (prefix / "correlation.json").write_text("{}", encoding="utf-8")

    _write_run_state(prefix, [
        {"stage": "describe", "model": "qwen3.5-0.8b", "status": "ok",
         "argv": "describe --model qwen3.5-0.8b"},
        {"stage": "mask", "model": "qwen3.5-0.8b", "status": "ok"},
        {"stage": "case-study", "model": "qwen3.5-0.8b", "status": "ok",
         "argv": "case-study --model qwen3.5-0.8b"},
        # ok, but `masking_curve.json` is not in this model's directory -> re-run
        {"stage": "mask", "model": "qwen3-0.6b", "status": "ok"},
        {"stage": "figures", "model": None, "status": "ok",
         "argv": "figures --runs ds/a ds/b"},
        {"stage": "detect", "model": "qwen3-0.6b", "status": "failed"},
        {"stage": "compare", "model": None, "status": "running"},
    ])
    assert driver.completed_stages(prefix) == {
        ("describe", "qwen3.5-0.8b"): "describe --model qwen3.5-0.8b",
        ("mask", "qwen3.5-0.8b"): None,          # recorded before the fingerprint existed
        ("case-study", "qwen3.5-0.8b"): "case-study --model qwen3.5-0.8b",
        ("figures", None): "figures --runs ds/a ds/b",
    }

    # The log is append-only, so the *last* status for a pair is the one that counts: a
    # stage that succeeded and then failed on a re-run must not be skipped.
    _write_run_state(prefix, [
        {"stage": "mask", "model": "qwen3.5-0.8b", "status": "ok"},
        {"stage": "mask", "model": "qwen3.5-0.8b", "status": "failed"},
    ])
    assert driver.completed_stages(prefix) == {}

    # A state file that cannot be read -- or is not shaped like the driver's -- means
    # "skip nothing", never "skip everything".
    (prefix / "run_state.json").write_text("{not json", encoding="utf-8")
    assert driver.completed_stages(prefix) == {}
    (prefix / "run_state.json").write_text("[]", encoding="utf-8")
    assert driver.completed_stages(prefix) == {}
    (prefix / "run_state.json").write_text('{"stages": {"mask": "ok"}}', encoding="utf-8")
    assert driver.completed_stages(prefix) == {}
    (prefix / "run_state.json").unlink()
    assert driver.completed_stages(prefix) == {}

    # `detect` is resumed on the `.npz` the ablations load, not on a `.json` sidecar that
    # can survive without it (the two are written together, but only the npz is read).
    (prefix / "qwen3-0.6b" / "scores_next_step.npz").unlink()
    _write_run_state(prefix, [{"stage": "detect", "model": "qwen3-0.6b", "status": "ok"}])
    assert driver.completed_stages(prefix) == {}


def test_main_resume_actually_skips_the_stage(driver, tmp_path, monkeypatch, capsys):
    """The skip must reach `main`, and only when the flag is passed."""
    import json

    prefix = tmp_path / "ds"
    (prefix / "qwen3-0.6b").mkdir(parents=True)
    (prefix / "qwen3-0.6b" / "masking_curve.json").write_text("{}", encoding="utf-8")
    _write_run_state(prefix, [{"stage": "mask", "model": "qwen3-0.6b", "status": "ok"}])

    monkeypatch.setattr(driver, "report_environment", lambda: None)
    monkeypatch.setattr(driver, "prepare_models", lambda args: None)
    monkeypatch.setattr(driver, "code_sha256", lambda *roots: "deadbeef")
    ran: list[str] = []
    monkeypatch.setattr(driver, "run_cli", lambda stage, argv: ran.append(stage))

    common = ["--weights", "/w", "--models", "qwen3-0.6b", "--stages", "mask",
              "--out-prefix", str(prefix)]
    assert driver.main([*common, "--resume"]) == 0
    assert ran == [], "a finished stage was paid for a second time"
    out = capsys.readouterr().out
    assert "skipping mask" in out
    assert "predates the argv fingerprint" in out, (
        "a record without an argv must say why it was trusted"
    )

    # Same prefix, no flag: it runs.  The skip is the flag's doing, not a side effect of
    # the state file merely existing.
    assert driver.main(common) == 0
    assert ran == ["mask"]

    # A record for the *same* stage with a *different* command line is not a skip: the
    # tree would otherwise mix two grids and report success (the reason the fingerprint
    # exists).  The re-run records the command it actually ran.
    ran.clear()
    fingerprint = driver.stage_fingerprint(
        driver.stage_argv("mask", profile="laptop", models=["qwen3-0.6b"],
                          prefix=prefix, seed=0)[0])
    _write_run_state(prefix, [{"stage": "mask", "model": "qwen3-0.6b", "status": "ok",
                               "argv": "mask --model qwen3-0.6b --lengths 1024"}])
    assert driver.main([*common, "--resume"]) == 0
    assert ran == ["mask"], "an artifact from another grid was kept"
    assert "different command line" in capsys.readouterr().out
    state = json.loads((prefix / "run_state.json").read_text(encoding="utf-8"))
    assert state["stages"][-1]["argv"] == fingerprint

    # ... and with the matching fingerprint it is skipped again.
    ran.clear()
    assert driver.main([*common, "--resume"]) == 0
    assert ran == []
    assert "same command line" in capsys.readouterr().out

    # Same prefix, no flag: it runs.  The skip is the flag's doing, not a side effect of
    # the state file merely existing.
    assert driver.main(common) == 0
    assert ran == ["mask"]
