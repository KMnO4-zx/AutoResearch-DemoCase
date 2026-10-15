"""Plot an AutoResearch series from local results and keep/discard decisions."""

import argparse
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
MAX_TRIALS = 10
METRIC = "eval/pass_at_3"


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_run(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("Use a run ID, not a directory path.")
    result = read_json(root / "runs" / run_id / "result.json")
    if result["run_id"] != run_id:
        raise ValueError(f"Run ID does not match its artifact: {run_id}")
    return result


def score(result):
    metrics = result.get("metrics", {})
    value = metrics.get(METRIC)
    if (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            or metrics.get("eval/problems") != 30 or metrics.get("eval/generations") != 90):
        raise ValueError(f"Missing or invalid full AIME25 score: {result['run_id']}")
    return float(value)


def commit_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Record a full 40-character Git commit SHA.")
    return value


def load_progress(root=ROOT, baseline_id=None):
    root = Path(root)
    if baseline_id is None:
        best_path = root / "best.json"
        if not best_path.exists():
            raise ValueError("No research series yet. Provide --baseline RUN_ID or record best.json after a baseline.")
        baseline_id = read_json(best_path)["baseline_run_id"]
    baseline = read_run(root, baseline_id)
    if (baseline["status"] != "complete" or baseline["kind"] != "experiment"
            or baseline["experiment"]["mode"] != "prompt" or baseline["model_path"] is not None
            or baseline["base_model"] != "Qwen/Qwen3.5-4B"):
        raise ValueError("Select a completed prompt-only 4B base-model baseline.")
    protocol = baseline["protocol"]
    if (protocol.get("rows"), protocol.get("n"), protocol.get("max_tokens")) != (30, 3, 1024):
        raise ValueError("Baseline must use all 30 AIME25 problems, pass@3, and 1024 output tokens.")
    if type(protocol.get("enable_thinking")) is not bool:
        raise ValueError("Baseline must record its evaluation thinking setting.")
    best_score = score(baseline)
    best_commit = commit_sha(baseline["code_commit"])
    points = [{"trial": 0, "run_id": baseline_id, "decision": "baseline", "mode": "prompt",
               "score": best_score, "best_score": best_score, "commit": best_commit,
               "label": "baseline", "reason": ""}]
    decisions_path = root / "decisions.jsonl"
    decisions = []
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each decision must be a JSON object.")
                if row.get("baseline_run_id") == baseline_id:
                    decisions.append(row)
    if len(decisions) > MAX_TRIALS:
        raise ValueError(f"A research series permits at most {MAX_TRIALS} candidate attempts.")
    seen = {baseline_id}
    for index, decision in enumerate(decisions, start=1):
        if type(decision.get("trial")) is not int or decision["trial"] != index:
            raise ValueError("Candidate trial numbers must be consecutive, starting at 1.")
        outcome = decision.get("decision")
        if outcome not in {"keep", "discard", "failed"}:
            raise ValueError(f"Unknown candidate decision: {outcome}")
        if commit_sha(decision.get("parent_commit")) != best_commit:
            raise ValueError(f"Trial {index} must start from the latest accepted commit.")
        selected_commit = commit_sha(decision.get("commit"))
        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Trial {index} requires a decision reason.")
        run_id = decision.get("run_id")
        candidate_score, mode = None, None
        label = decision.get("label", "")
        if not isinstance(label, str):
            raise ValueError("Decision labels must be strings.")
        if run_id is None:
            if outcome != "failed":
                raise ValueError("Only a failed attempt may omit its run artifact.")
        else:
            if run_id in seen:
                raise ValueError("Each candidate run may appear only once in a series.")
            seen.add(run_id)
            result = read_run(root, run_id)
            if result["kind"] != "experiment":
                raise ValueError("Confirmations are separate; do not count them as candidates.")
            for key in ("base_model", "protocol", "data_sha256", "prepare_sha256", "evaluation_sha256"):
                if result[key] != baseline[key]:
                    raise ValueError(f"Trial {index} changes the comparison's {key}.")
            if result["code_commit"] != best_commit:
                raise ValueError(f"Trial {index} run was not based on the latest accepted commit.")
            mode = result["experiment"]["mode"]
            label = label or result["experiment"].get("hypothesis") or result["name"]
            if outcome == "failed":
                if result["status"] != "failed":
                    raise ValueError("Failed decisions require a failed result.")
            else:
                if result["status"] != "complete":
                    raise ValueError("A kept/discarded candidate requires a completed result.")
                candidate_score = score(result)
        if outcome == "keep":
            if candidate_score <= best_score or selected_commit == best_commit:
                raise ValueError("A keep requires a strictly higher score and a new commit.")
            best_score, best_commit = candidate_score, selected_commit
        elif selected_commit != best_commit:
            raise ValueError("Discarded/failed attempts must retain the latest accepted commit.")
        points.append({"trial": index, "run_id": run_id, "decision": outcome, "mode": mode,
                       "score": candidate_score, "best_score": best_score, "commit": selected_commit,
                       "parent_commit": decision["parent_commit"], "label": label, "reason": reason,
                       "confirmation_run_ids": decision.get("confirmation_run_ids", [])})
    selected = next(point for point in reversed(points) if point["decision"] in {"baseline", "keep"})
    return {"baseline_run_id": baseline_id, "base_model": baseline["base_model"], "metric": METRIC,
            "protocol": protocol, "points": points, "best_run_id": selected["run_id"],
            "best_commit": best_commit, "best_score": best_score}


def plot_progress(progress, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ("PingFang SC", "Heiti TC", "Songti SC", "Noto Sans CJK SC", "Arial Unicode MS")
             if name in available]
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Use a .png output path; a matching SVG and data JSON are also saved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    points = progress["points"]
    kept = [point for point in points if point["decision"] == "keep"]
    discarded = [point for point in points if point["decision"] == "discard"]
    failed = [point for point in points if point["decision"] == "failed"]
    green = "#2ecc71"
    with plt.rc_context({"font.family": fonts + ["DejaVu Sans"], "font.size": 11, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
        ax.grid(color="#e8e8e8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.scatter([p["trial"] for p in discarded], [p["score"] for p in discarded],
                   color="#c7c7c7", alpha=0.55, s=38, label="Discarded", zorder=3)
        accepted = [points[0]] + kept
        ax.scatter([p["trial"] for p in accepted], [p["score"] for p in accepted],
                   color=green, edgecolors="#227b45", linewidths=1.4, s=80,
                   label="Baseline / kept", zorder=4)
        ax.step([p["trial"] for p in points], [p["best_score"] for p in points], where="post",
                color=green, alpha=0.8, linewidth=2.4, label="Running best (kept)", zorder=2)
        if failed:
            ax.scatter([p["trial"] for p in failed], [0.035] * len(failed), transform=ax.get_xaxis_transform(),
                       color="#d45a5a", marker="x", s=50, label="Failed (no score)", zorder=4)
        for index, point in enumerate(accepted):
            label = point["label"]
            if point["decision"] == "keep":
                label = f"{point['mode']}: {label}"
            label = label if len(label) <= 64 else label[:61] + "..."
            at_end = point["trial"] >= max(1, points[-1]["trial"] - 1)
            ax.annotate(label, (point["trial"], point["score"]),
                        xytext=(-10 if at_end else 10, 16 + 16 * (index % 2)), textcoords="offset points",
                        ha="right" if at_end else "left", rotation=16, color="#299c57", fontsize=10)
        values = [point["score"] for point in points if point["score"] is not None]
        ax.set_ylim(max(0, min(values) - 0.09), min(1.08, max(values) + 0.18))
        ax.set_xlim(-0.35, max(1, points[-1]["trial"]) + 0.55)
        ax.set_xticks([point["trial"] for point in points])
        ax.set_xlabel("Candidate attempt # (0 = baseline)")
        ax.set_ylabel("AIME25 pass@3 (higher is better)")
        ax.set_title(f"AutoResearch-PyTRIO Progress: {len(points) - 1} Attempts, {len(kept)} Kept Improvements", pad=18)
        protocol = progress["protocol"]
        ax.text(0.985, 0.985, f"Qwen3.5-4B | thinking {'on' if protocol['enable_thinking'] else 'off'} | seed {protocol['seed']}",
                transform=ax.transAxes, ha="right", va="top", color="#777777", fontsize=9)
        ax.text(0.985, 0.09, f"Best: {progress['best_commit'][:8]} / pass@3={progress['best_score']:.3f}",
                transform=ax.transAxes, ha="right", color="#777777", fontsize=9)
        ax.legend(loc="upper left", framealpha=0.95)
        try:
            fig.savefig(output, dpi=180)
            fig.savefig(output.with_suffix(".svg"))
        finally:
            plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    return [output, output.with_suffix(".svg"), output.with_suffix(".json")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing runs and decisions.jsonl.")
    parser.add_argument("--baseline", help="Baseline run ID. Default: best.json baseline_run_id.")
    parser.add_argument("--output", type=Path, help="PNG path. Default: ROOT/runs/progress.png.")
    args = parser.parse_args()
    try:
        progress = load_progress(args.root, args.baseline)
        outputs = plot_progress(progress, args.output or args.root / "runs" / "progress.png")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
