"""Validate saved real-checkpoint smoke artifacts, not model answer quality."""
import argparse
import json
import math
from pathlib import Path


METRICS = {"legacy", "needle_token_multiset_v1", "needle_attention_mass_v1"}
HEADS = {f"{layer}-{head}" for layer in range(34) for head in range(32)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(run_root):
    detection = Path(run_root) / "detection"
    manifest = json.loads((detection / "run.json").read_text())
    require(manifest["complete"], "Detection is incomplete")
    require(manifest["adapter"] == "ministral3_8b_no_thinking", "Wrong adapter")
    require(manifest["total_cases"] == manifest["completed_cases"] == 3, "Expected three cases")
    require(manifest["lengths"] == [1000] and manifest["depths"] == [50], "Wrong smoke grid")
    require(manifest["attention_scope"] == "all_decode_tokens", "Wrong attention scope")
    require(set(manifest["retrieval_metrics"]) == METRICS, "Missing metrics")
    summaries = []
    positive_mass = False
    for path in sorted((detection / "results").glob("*_results.json")):
        row = json.loads(path.read_text())
        experiment = row["experiment"]
        if experiment["run_id"] != manifest["run_id"]:
            continue
        require(0 < len(row["generated_token_ids"]) <= 128, f"Invalid token count: {path}")
        require(row["model_response"].strip() and row["final_answer_started"], f"Empty answer: {path}")
        require(row["finish_reason"] in {"eos", "max_new_tokens"}, f"Invalid finish reason: {path}")
        require(math.isfinite(row["score"]), f"Invalid ROUGE: {path}")
        scores = experiment["retrieval_scores"]
        require(set(scores) == METRICS, f"Missing per-case metrics: {path}")
        for metric, values in scores.items():
            require(set(values) == HEADS, f"Expected 1088 heads: {path}, {metric}")
            require(all(math.isfinite(v) and v >= 0 for v in values.values()), f"Invalid scores: {path}, {metric}")
            if metric != "legacy":
                require(all(v <= 1.000001 for v in values.values()), f"Score exceeds one: {path}, {metric}")
        qualifying = experiment["retrieval_qualifying_steps"]["needle_attention_mass_v1"]
        positive_mass |= qualifying > 0 and max(scores["needle_attention_mass_v1"].values()) > 0
        summaries.append({"case_id": row["case_id"], "score": row["score"],
                          "tokens": len(row["generated_token_ids"]), "finish_reason": row["finish_reason"],
                          "qualifying_steps": qualifying, "response": row["model_response"]})
    require({row["case_id"] for row in summaries} == {"detect-1", "detect-2", "detect-3"}
            and len(summaries) == 3, "Missing or duplicate cases")
    require(positive_mass, "No positive needle attention mass observed")
    aggregation = json.loads((detection / "aggregation" / "run.json").read_text())
    require(aggregation["source_run_id"] == manifest["run_id"]
            and aggregation["selected_case_count"] == 3, "Incomplete aggregation")
    for metric in METRICS:
        history = json.loads((detection / "aggregation" / aggregation["head_score_files"][metric]).read_text())
        require(set(history) == HEADS and all(len(values) == 3 for values in history.values()),
                f"Invalid aggregated history: {metric}")
    report = {"passed": True, "head_count": len(HEADS), "cases": summaries}
    (Path(run_root) / "smoke_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().run), indent=2, ensure_ascii=False), flush=True)
