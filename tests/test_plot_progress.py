"""Local fixtures for progress accounting and chart rendering; no model calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import plot_progress as p


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root):
    """A test-only series with prompt improvements, rejection, and two failures."""
    commits = [digit * 40 for digit in "123"]
    baseline = {"run_id": "baseline", "name": "baseline", "kind": "experiment", "status": "complete",
                "base_model": "Qwen/Qwen3.5-4B", "model_path": None, "code_commit": commits[0],
                "prepare_sha256": "fixed-utilities", "evaluation_sha256": "fixed-evaluation",
                "data_sha256": {"aime25.jsonl": "fixed-data"},
                "experiment": {"mode": "prompt", "hypothesis": "Initial prompt"},
                "protocol": {"rows": 30, "n": 3, "max_tokens": 1024, "seed": 42, "enable_thinking": False},
                "metrics": {"eval/pass_at_3": 12 / 30, "eval/problems": 30, "eval/generations": 90}}
    write_json(root / "runs" / "baseline" / "result.json", baseline)
    decisions = []
    parent = commits[0]
    trials = [("keep", 15 / 30, "prompt", commits[1], "Concise reasoning"),
              ("discard", 17 / 30, "grpo", commits[1], "Gain failed paired confirmation"),
              ("failed", None, "opd", commits[1], "Service request failed"),
              ("keep", 18 / 30, "prompt", commits[2], "检查答案后再输出"),
              ("discard", 18 / 30, "sft", commits[2], "Tie with current best"),
              ("failed", None, None, commits[2], "Configuration failure before run creation")]
    for index, (outcome, value, mode, commit, label) in enumerate(trials, start=1):
        run_id = f"trial-{index}" if mode else None
        if run_id:
            result = deepcopy(baseline)
            result.update({"run_id": run_id, "name": run_id, "code_commit": parent,
                           "status": "failed" if outcome == "failed" else "complete",
                           "experiment": {"mode": mode, "hypothesis": label}})
            if value is None:
                result.pop("metrics")
            else:
                result["metrics"]["eval/pass_at_3"] = value
            write_json(root / "runs" / run_id / "result.json", result)
        decisions.append({"baseline_run_id": "baseline", "trial": index, "run_id": run_id,
                          "decision": outcome, "parent_commit": parent, "commit": commit,
                          "label": label, "reason": label})
        parent = commit
    (root / "decisions.jsonl").write_text("\n".join(json.dumps(row) for row in decisions) + "\n", encoding="utf-8")
    write_json(root / "best.json", {"baseline_run_id": "baseline", "run_id": "trial-4", "commit": commits[2]})
    return baseline, decisions


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline, self.decisions = make_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def save_decisions(self, rows):
        (self.root / "decisions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_running_best_follows_acceptance_and_counts_failed_attempts(self):
        progress = p.load_progress(self.root)
        self.assertEqual([point["trial"] for point in progress["points"]], list(range(7)))
        self.assertEqual([point["best_score"] for point in progress["points"]], [0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
        self.assertEqual(progress["best_commit"], "3" * 40)
        self.assertEqual(progress["best_run_id"], "trial-4")
        self.assertEqual(progress["points"][1]["mode"], "prompt")
        self.assertIsNone(progress["points"][3]["score"])
        self.assertIsNone(progress["points"][6]["run_id"])

    def test_mixed_protocol_models_data_or_utilities_are_rejected(self):
        path = self.root / "runs" / "trial-1" / "result.json"
        original = p.read_json(path)
        variants = [{"protocol": original["protocol"] | {"enable_thinking": True}},
                    {"protocol": original["protocol"] | {"seed": 1042}},
                    {"base_model": "Qwen/Qwen3.5-9B"}, {"data_sha256": {}},
                    {"prepare_sha256": "changed"}, {"evaluation_sha256": "changed"}]
        for changes in variants:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "changes the comparison"):
                write_json(path, original | changes)
                p.load_progress(self.root)

    def test_keep_requires_strict_score_improvement_and_a_new_commit(self):
        path = self.root / "runs" / "trial-1" / "result.json"
        original = p.read_json(path)
        write_json(path, original | {"metrics": original["metrics"] | {"eval/pass_at_3": 0.4}})
        with self.assertRaisesRegex(ValueError, "strictly higher"):
            p.load_progress(self.root)
        write_json(path, original)
        self.save_decisions([self.decisions[0] | {"commit": "1" * 40}])
        with self.assertRaisesRegex(ValueError, "new commit"):
            p.load_progress(self.root)

    def test_stale_parent_and_discard_that_changes_commit_are_rejected(self):
        for field in ("parent_commit", "commit"):
            modified = deepcopy(self.decisions)
            modified[1][field] = "1" * 40
            self.save_decisions(modified)
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.load_progress(self.root)

    def test_invalid_and_partial_scores_are_rejected(self):
        path = self.root / "runs" / "trial-1" / "result.json"
        original = p.read_json(path)
        variants = [{"eval/pass_at_3": float("nan")}, {"eval/pass_at_3": True},
                    {"eval/pass_at_3": 1.1}, {"eval/generations": 89}, {"eval/problems": 29}]
        for metrics in variants:
            write_json(path, original | {"metrics": original["metrics"] | metrics})
            with self.subTest(metrics=metrics), self.assertRaisesRegex(ValueError, "full AIME25 score"):
                p.load_progress(self.root)

    def test_confirmations_cannot_be_counted_as_candidate_points(self):
        path = self.root / "runs" / "trial-1" / "result.json"
        write_json(path, p.read_json(path) | {"kind": "reevaluation"})
        with self.assertRaisesRegex(ValueError, "Confirmations are separate"):
            p.load_progress(self.root)

    def test_limit_and_consecutive_numbering_include_failures(self):
        self.save_decisions([{"baseline_run_id": "baseline", "trial": index, "run_id": None,
                              "decision": "failed", "parent_commit": "1" * 40, "commit": "1" * 40,
                              "reason": "Failed before artifact"} for index in range(1, 12)])
        with self.assertRaisesRegex(ValueError, "at most 10"):
            p.load_progress(self.root)
        self.save_decisions([self.decisions[0] | {"trial": 2}])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            p.load_progress(self.root)

    def test_missing_artifact_requires_failed_decision(self):
        self.save_decisions([self.decisions[0] | {"run_id": None}])
        with self.assertRaisesRegex(ValueError, "Only a failed attempt"):
            p.load_progress(self.root)

    def test_explicit_baseline_supports_a_series_before_any_candidate(self):
        (self.root / "decisions.jsonl").unlink()
        (self.root / "best.json").unlink()
        progress = p.load_progress(self.root, "baseline")
        self.assertEqual(len(progress["points"]), 1)
        self.assertEqual(progress["best_score"], 0.4)
        with tempfile.TemporaryDirectory() as empty, self.assertRaisesRegex(ValueError, "No research series"):
            p.load_progress(Path(empty))

    def test_render_exports_png_svg_and_auditable_json_with_chinese_labels(self):
        progress = p.load_progress(self.root)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            outputs = p.plot_progress(progress, self.root / "chart" / "progress.png")
        self.assertEqual(outputs[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("<svg", outputs[1].read_text())
        self.assertEqual(p.read_json(outputs[2]), progress)


if __name__ == "__main__":
    unittest.main()
