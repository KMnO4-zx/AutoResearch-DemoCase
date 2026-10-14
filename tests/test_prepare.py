"""Local contract tests. Every model-service operation is replaced with a fake."""

import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import prepare as p


def config(**changes):
    return replace(p.read_experiment(), **changes)


def rows(count, prefix="train"):
    return [{"id": str(i), "problem": f"{prefix} question {i}; target=70", "answer": "70"}
            for i in range(count)]


class FakeTokenizer:
    eos_token = "<END>"

    def __init__(self):
        self.thinking_settings = []

    def apply_chat_template(self, messages, **kwargs):
        thinking = kwargs["enable_thinking"]
        assert type(thinking) is bool
        self.thinking_settings.append(thinking)
        return messages[0]["content"] + ("\n<think>" if thinking else "\n<think></think>")

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()

    def get_vocab(self):
        return {str(index): index for index in range(256)}


class RemoteFuture:
    def __init__(self, result, callback=None, error=None):
        self.result, self.callback, self.error = result, callback, error

    def __await__(self):
        async def finish():
            if self.error:
                raise self.error
            if self.callback:
                self.callback()
            return self.result
        return finish().__await__()


class FakeSampler:
    def __init__(self, mixed=True):
        self.mixed = mixed
        self.active = self.peak = 0
        self.calls = []
        self.tokenizer = FakeTokenizer()
        self.prompts = []
        self.fail_seed = None
        self.missing_sample = False
        self.over_budget = False

    def get_tokenizer(self):
        return self.tokenizer

    async def sample_async(self, prompt, num_samples, sampling_params, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append((num_samples, sampling_params))
        self.prompts.append(prompt.to_ints())
        try:
            await asyncio.sleep(0.001)
            if sampling_params.seed == self.fail_seed:
                raise RuntimeError("Simulated sampling failure")
            sequences = []
            for index in range(num_samples - int(self.missing_sample)):
                answer = "070" if self.mixed and index == 0 else "71"
                text = "Reasoning. \\boxed{" + answer + "}"
                tokens = list(text.encode())
                if self.over_budget:
                    tokens = [1] * (p.MAX_TOKENS + 1)
                sequences.append(SimpleNamespace(tokens=tokens, text=text,
                                                 logprobs=[-0.5] * len(tokens), stop_reason="stop"))
            return SimpleNamespace(sequences=sequences, input_tokens=len(prompt))
        finally:
            self.active -= 1


class FakeTraining:
    def __init__(self, mixed=True):
        self.sampler = FakeSampler(mixed)
        self.refreshes = self.backward_submissions = self.backward_completed = self.updates = 0
        self.backward_failure = False
        self.loss_fns = []
        self.datums = []

    def get_tokenizer(self):
        return self.sampler.get_tokenizer()

    async def save_weights_and_get_sampling_client_async(self):
        self.refreshes += 1
        return self.sampler

    async def forward_backward_async(self, data, loss_fn):
        assert loss_fn in {"importance_sampling", "cross_entropy"}
        self.loss_fns.append(loss_fn)
        self.datums.extend(data)
        assert len(data) * max(len(d.model_input) for d in data) <= p.MICRO_BATCH_PADDED_TOKENS
        self.backward_submissions += 1
        def completed():
            self.backward_completed += 1
        return RemoteFuture(SimpleNamespace(metrics={"loss": 0.5}), callback=completed,
                            error=RuntimeError("Simulated backward failure") if self.backward_failure else None)

    async def optim_step_async(self, params):
        assert self.backward_completed == self.backward_submissions
        def completed():
            self.updates += 1
        return RemoteFuture(None, callback=completed)

    async def save_weights_for_sampler_async(self, name):
        return RemoteFuture(SimpleNamespace(path=f"trio://fake/{name}"))


class FakeService:
    def __init__(self):
        self.training_clients = []
        self.sampler = FakeSampler()
        self.sampling_arguments = []
        self.teacher = FakeTeacher()
        self.supported_models = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"]

    def get_supported_models(self):
        return self.supported_models

    async def create_lora_training_client_async(self, **kwargs):
        # Candidates always start fresh; no previous weights or training state.
        assert set(kwargs) == {"base_model", "rank", "seed"}
        assert kwargs["base_model"] == p.TARGET_MODEL
        client = FakeTraining()
        self.training_clients.append(client)
        return client

    async def create_sampling_client_async(self, **kwargs):
        self.sampling_arguments.append(kwargs)
        return self.teacher if kwargs["base_model"] == "Qwen/Qwen3.8-27B" else self.sampler


class FakeTeacher:
    def __init__(self):
        self.calls = []
        self.active = self.peak = 0
        self.invalid_logprobs = False
        self.incompatible_vocab = False

    def get_tokenizer(self):
        tokenizer = FakeTokenizer()
        if self.incompatible_vocab:
            tokenizer.get_vocab = lambda: {"different": 1}
        return tokenizer

    async def compute_logprobs_async(self, prompt):
        ids = prompt.to_ints()
        self.calls.append(ids)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001)
            return [None] + [-1.0] * (len(ids) - 2) + [None if self.invalid_logprobs else -1.0]
        finally:
            self.active -= 1


class DataAndConfigTests(unittest.TestCase):
    def test_all_training_modes_use_fixed_student_and_catalog_teachers(self):
        for mode in ("prompt", "grpo", "sft", "opd"):
            config(mode=mode).validate()
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "target is fixed"):
                config(mode=mode, base_model="Qwen/Qwen3.5-9B").validate()
        service = FakeService()
        service.supported_models.append("vendor/future-teacher")
        p.validate_supported_models(service, config(mode="opd", teacher_base_model="vendor/future-teacher"),
                                    require_teacher=True)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            p.validate_supported_models(service, config(mode="opd", teacher_base_model="vendor/unavailable-teacher"),
                                        require_teacher=True)
        config(mode="opd", group_size=1).validate()
        with self.assertRaises(ValueError):
            config(mode="grpo", group_size=1).validate()

    def test_sft_target_prompt_mask_and_eos(self):
        sample = p.sft_sample(FakeTokenizer(), rows(1)[0], config(mode="sft"))
        self.assertEqual(sample["tokens"][-len(b"<END>"):], list(b"<END>"))
        datum = p.sft_datum(sample)
        padding = len(sample["prompt_tokens"]) - 1
        weights = datum.loss_fn_inputs["weights"].to_numpy()
        np.testing.assert_array_equal(weights[:padding], np.zeros(padding))
        np.testing.assert_array_equal(weights[padding:], np.ones(len(sample["tokens"])))
        self.assertEqual(len(datum.model_input), len(weights))
        with self.assertRaises(ValueError):
            p.sft_sample(FakeTokenizer(), rows(1)[0] | {"solution": r"Wrong \boxed{71}"}, config(mode="sft"))

    def test_opd_teacher_may_have_a_compatible_vocabulary_superset(self):
        teacher = FakeTokenizer()
        teacher.get_vocab = lambda: FakeTokenizer().get_vocab() | {"extra-teacher-token": 256}
        p.check_opd_tokenizers(FakeTokenizer(), teacher)

    def test_step_budget_rejects_invalid_configuration(self):
        for steps in (21, -1, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                config(steps=steps).validate()

    def test_configuration_read_does_not_execute_service_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "experiment.py"
            source.write_text("raise RuntimeError('must not execute')\nEXPERIMENT = " +
                              repr(p.read_experiment().__dict__))
            with patch.object(p.trio, "ServiceClient", side_effect=AssertionError("No model service")):
                self.assertEqual(p.read_experiment(source).steps, 20)
                source.write_text(source.read_text().replace("'steps': 20", "'steps': 19"))
                self.assertEqual(p.read_experiment(source).steps, 19)

    def test_last_boxed_answer_and_integer_normalization(self):
        self.assertEqual(p.grade(r"Earlier \boxed{71}, final \boxed{\text{070}}", "70"), (True, True))
        self.assertEqual(p.grade(r"Earlier \boxed{70}, final \boxed{", "70"), (False, False))
        self.assertEqual(p.grade("The answer might be 70", "70"), (False, False))

    def test_data_deduplication_zero_answer_conflicts_and_overlap(self):
        def raw(question, answer):
            return {"prompt": [{"content": p.DAPO_PREFIX + question + p.DAPO_SUFFIX}],
                    "reward_model": {"ground_truth": answer}}
        data = [raw("overlap", "70"), raw("zero", 0), raw("zero", "0"),
                raw("conflict", "6"), raw("conflict", "64"),
                raw("negative", "-3"), raw("fraction", "1/2")]
        selected, counts = p.collect_train(data, [{"problem": "overlap"}], 2)
        self.assertEqual({row["answer"] for row in selected}, {"0", "-3"})
        self.assertEqual(counts["conflicting_questions"], 1)
        self.assertEqual(counts["eval_overlap"], 1)
        self.assertEqual(counts["invalid"], 1)

    def test_full_scan_excludes_conflict_after_requested_subset_size(self):
        data = [{"prompt": [{"content": q}], "reward_model": {"ground_truth": a}}
                for q, a in [("first", "1"), ("second", "2"), ("first", "9")]]
        selected, counts = p.collect_train(data, [], 1)
        self.assertEqual(selected[0]["problem"], "second")
        self.assertEqual(counts["scanned"], 3)

    def test_grpo_autoregressive_shift_and_prompt_mask(self):
        sample = {"prompt_tokens": [10, 11, 12], "tokens": [20, 21],
                  "logprobs": [-0.4, -0.8], "advantage": 0.75}
        datum = p.grpo_datum(sample, scale=0.25)
        self.assertEqual(datum.model_input.to_ints(), [10, 11, 12, 20])
        np.testing.assert_array_equal(datum.loss_fn_inputs["target_tokens"].to_numpy(), [0, 0, 20, 21])
        np.testing.assert_allclose(datum.loss_fn_inputs["logprobs"].to_numpy(), [0, 0, -0.4, -0.8])
        np.testing.assert_allclose(datum.loss_fn_inputs["advantages"].to_numpy(), [0, 0, 0.1875, 0.1875])
        with self.assertRaises(ValueError):
            p.grpo_datum(sample | {"logprobs": [None, -0.8]})

    def test_micro_batch_padded_area_is_bounded(self):
        samples = [{"prompt_tokens": [1] * length, "tokens": [2] * 1024}
                   for length in [2048] * 16]
        batches = p.pack_samples(samples)
        self.assertEqual(sum(map(len, batches)), 16)
        for batch in batches:
            self.assertLessEqual(len(batch) * max(len(s["prompt_tokens"]) + len(s["tokens"]) - 1
                                                for s in batch), p.MICRO_BATCH_PADDED_TOKENS)


class AsyncContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_opd_scores_the_actual_student_tokens_and_aligns_advantages(self):
        teacher = FakeTeacher()
        sample = {"prompt_tokens": [10, 11], "tokens": [20, 21, 22], "logprobs": [-0.5, -2.0, -1.0]}
        kls = await p.opd_advantages(teacher, sample, 2.0, asyncio.Semaphore(1))
        self.assertEqual(teacher.calls, [[10, 11, 20, 21, 22]])
        self.assertEqual(kls, [0.5, -1.0, 0.0])
        np.testing.assert_array_equal(p.grpo_datum(sample).loss_fn_inputs["advantages"].to_numpy(),
                                      [0.0, -1.0, 2.0, 0.0])

    async def test_opd_has_teacher_signal_even_when_all_answers_are_wrong(self):
        training, teacher, history = FakeTraining(mixed=False), FakeTeacher(), []
        result = await p.train_model(training, FakeTokenizer(), rows(80),
                                     config(mode="opd", group_size=1, steps=20),
                                     lambda metrics, step: history.append(metrics), teacher=teacher)
        self.assertEqual((result["optimizer_steps"], training.updates, training.refreshes), (20, 20, 20))
        self.assertEqual(history[-1]["train/reward"], 0)
        self.assertEqual(history[-1]["opd/reverse_kl_mean"], 0.5)
        self.assertEqual(len(teacher.calls), 80)
        self.assertLessEqual(teacher.peak, 4)

    async def test_opd_bad_token_mapping_and_teacher_logprobs_fail_before_updates(self):
        for failure in ("incompatible_vocab", "invalid_logprobs"):
            training, teacher = FakeTraining(), FakeTeacher()
            setattr(teacher, failure, True)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                await p.train_model(training, FakeTokenizer(), rows(80), config(mode="opd"),
                                     lambda *_: None, teacher=teacher)
            self.assertEqual(training.updates, 0)
            self.assertEqual(training.backward_submissions, 0)

    async def test_sft_uses_cross_entropy_without_rollout_and_obeys_twenty_steps(self):
        training, history = FakeTraining(), []
        result = await p.train_model(training, FakeTokenizer(), rows(80), config(mode="sft", steps=20),
                                     lambda metrics, step: history.append(metrics))
        self.assertEqual((result["optimizer_steps"], training.updates), (20, 20))
        self.assertEqual(training.refreshes, 0)
        self.assertEqual(training.sampler.calls, [])
        self.assertEqual((result["training_rounds"], result["rollout_rounds"]), (20, 0))
        self.assertEqual(set(training.loss_fns), {"cross_entropy"})
        self.assertEqual(history[-1]["sft/answer_only_fraction"], 1.0)
        self.assertNotIn("train/reward", history[-1])

    async def test_evaluation_pass_at_three_and_bounded_concurrency(self):
        sampler = FakeSampler()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary) / "generations.jsonl"
            metrics = await p.evaluate(sampler, FakeTokenizer(), rows(30, "aime"), config(), output)
            self.assertEqual(len(p.read_rows(output)), 30)
        self.assertEqual(metrics["eval/pass_at_3"], 1.0)
        self.assertAlmostEqual(metrics["eval/average_at_3"], 1 / 3)
        self.assertEqual(metrics["eval/generations"], 90)
        self.assertEqual(sampler.peak, 16)
        for n, params in sampler.calls:
            self.assertEqual((n, params.max_tokens, params.temperature), (3, 1024, 0.8))

    async def test_incomplete_group_and_over_budget_output_fail(self):
        for attribute in ("missing_sample", "over_budget"):
            sampler = FakeSampler()
            setattr(sampler, attribute, True)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                await p.sample_group(sampler, FakeTokenizer(), rows(1)[0], config(), 3,
                                     0.8, 42, asyncio.Semaphore(1))

    async def test_failed_evaluation_cancels_pending_tasks(self):
        sampler = FakeSampler()
        sampler.fail_seed = p.EVAL_SEED + 9
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary) / "generations.jsonl"
            with self.assertRaises(RuntimeError):
                await p.evaluate(sampler, FakeTokenizer(), rows(30, "aime"), config(), output)
            self.assertLess(len(p.read_rows(output)), 30)
        self.assertEqual(sampler.active, 0)

    async def test_twenty_updates_and_sampler_refresh(self):
        training = FakeTraining()
        metrics = []
        result = await p.train_model(training, FakeTokenizer(), rows(80), config(steps=20),
                                    lambda values, step: metrics.append(values))
        self.assertEqual((result["optimizer_steps"], training.updates, training.refreshes), (20, 20, 20))
        self.assertEqual(metrics[-1]["train/optimizer_steps"], 20)
        self.assertLessEqual(training.sampler.peak, 4)
        invalid = FakeTraining()
        with self.assertRaises(ValueError):
            await p.train_model(invalid, FakeTokenizer(), rows(80), config(steps=21), lambda *_: None)
        self.assertEqual(invalid.refreshes, 0)

    async def test_zero_signal_stops_without_optimizer_update(self):
        training = FakeTraining(mixed=False)
        result = await p.train_model(training, FakeTokenizer(), rows(80), config(), lambda *_: None)
        self.assertTrue(result["early_stop_no_signal"])
        self.assertEqual((result["rollout_rounds"], training.updates, training.backward_submissions), (3, 0, 0))

    async def test_backward_failure_never_submits_optimizer_update(self):
        training = FakeTraining()
        training.backward_failure = True
        with self.assertRaises(RuntimeError):
            await p.train_model(training, FakeTokenizer(), rows(80), config(), lambda *_: None)
        self.assertEqual(training.updates, 0)

    async def test_multiple_micro_batches_use_one_optimizer_update(self):
        training = FakeTraining()
        with patch.object(p, "MICRO_BATCH_PADDED_TOKENS", 800):
            result = await p.train_model(training, FakeTokenizer(), rows(80), config(steps=1), lambda *_: None)
        self.assertGreater(training.backward_submissions, 1)
        self.assertEqual(result["optimizer_steps"], 1)


class LocalWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.patches = [patch.object(p, "ROOT", self.root), patch.object(p, "DATA", self.data),
                        patch.object(p, "RUNS", self.root / "runs")]
        for item in self.patches:
            item.start()
        p.write_rows(self.data / "train.jsonl", rows(80))
        p.write_rows(self.data / "aime25.jsonl", rows(30, "aime"))
        p.write_json(self.data / "manifest.json", {
            "train_rows": 80, "eval_source": p.EVAL_SOURCE, "eval_revision": p.EVAL_REVISION,
            "sha256": {name: p.sha256(self.data / name) for name in ["train.jsonl", "aime25.jsonl"]},
        })
        (self.root / "experiment.py").write_text("EXPERIMENT = " + repr(config(steps=2).__dict__))
        self.service = FakeService()
        self.service_patch = patch.object(p.trio, "ServiceClient", return_value=self.service)
        self.service_patch.start()
        # Default arguments capture the production path; explicitly redirect the reader.
        original_reader = p.read_experiment
        self.reader_patch = patch.object(p, "read_experiment", side_effect=lambda **overrides:
                                         original_reader(self.root / "experiment.py", **overrides))
        self.reader_patch.start()

    def tearDown(self):
        self.reader_patch.stop()
        self.service_patch.stop()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def args(self, **changes):
        return SimpleNamespace(**({"command": "run", "mode": None, "name": "local-test",
                                   "base_model": p.TARGET_MODEL, "eval_seed": 42, "eval_thinking": None,
                                   "teacher_base_model": None,
                                   "swanlab_mode": "disabled", "swanlab_project": "local-validation",
                                   "swanlab_workspace": None} | changes))

    def results(self):
        return [json.loads(path.read_text()) for path in sorted((self.root / "runs").glob("*/result.json"))]

    async def test_opd_configuration_teacher_and_checkpoint_are_recorded(self):
        with redirect_stdout(io.StringIO()):
            await p.run_experiment(self.args(mode="opd"))
        result = self.results()[0]
        self.assertEqual(result["experiment"]["mode"], "opd")
        self.assertEqual(result["base_model"], p.TARGET_MODEL)
        self.assertEqual(result["experiment"]["teacher_base_model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(result["training"]["optimizer_steps"], 2)
        self.assertTrue(self.service.teacher.calls)

    async def test_non_target_student_is_rejected_before_service_creation(self):
        with self.assertRaisesRegex(ValueError, "target is fixed"):
            await p.run_experiment(self.args(base_model="Qwen/Qwen3.5-9B"))
        self.service_patch.target.ServiceClient.assert_not_called()

    async def test_evaluation_thinking_reaches_prompts_and_records_with_same_budget(self):
        for setting in (None, False, True):
            with self.subTest(setting=setting), redirect_stdout(io.StringIO()):
                name = f"thinking-{setting}"
                await p.run_experiment(self.args(name=name, eval_thinking=setting))
                result = next(row for row in self.results() if row["name"] == name)
                expected = setting is True
                self.assertIs(result["protocol"]["enable_thinking"], expected)
                self.assertEqual(self.service.sampler.tokenizer.thinking_settings[-30:], [expected] * 30)
                suffix = b"\n<think>" if expected else b"\n<think></think>"
                self.assertTrue(all(bytes(prompt).endswith(suffix) for prompt in self.service.sampler.prompts[-30:]))
                self.assertTrue(all(n == 3 and params.max_tokens == 1024
                                    for n, params in self.service.sampler.calls[-30:]))
                # This is an evaluation-only switch; training keeps its own template behavior.
                training = self.service.training_clients[-1]
                self.assertEqual(training.get_tokenizer().thinking_settings, [False] * 8)

    async def test_reevaluation_inherits_thinking_and_allows_explicit_override(self):
        with redirect_stdout(io.StringIO()):
            await p.run_experiment(self.args(mode="prompt", eval_thinking=True))
            original = self.results()[0]
            directory = self.root / "runs" / original["run_id"]
            for setting, expected in ((None, True), (False, False), (True, True)):
                name = f"confirm-{setting}"
                await p.run_experiment(self.args(command="evaluate", name=name, run=directory,
                                                 eval_seed=1042, eval_thinking=setting))
                result = next(row for row in self.results() if row["name"] == name)
                self.assertIs(result["protocol"]["enable_thinking"], expected)
                self.assertEqual(result["protocol"]["seed"], 1042)
                self.assertEqual(self.service.sampler.tokenizer.thinking_settings[-30:], [expected] * 30)
                self.assertEqual(result["source_run"], original["run_id"])
        self.assertEqual(self.service.training_clients, [])

    async def test_reevaluation_rejects_mismatched_saved_student(self):
        with redirect_stdout(io.StringIO()):
            await p.run_experiment(self.args(mode="prompt"))
        original = self.results()[0]
        directory = self.root / "runs" / original["run_id"]
        p.write_json(directory / "result.json", original | {"base_model": "Qwen/Qwen3.5-9B"})
        count = len(self.service.sampling_arguments)
        with self.assertRaisesRegex(ValueError, "fixed 4B target"):
            await p.run_experiment(self.args(command="evaluate", run=directory))
        self.assertEqual(len(self.service.sampling_arguments), count)

    async def test_unavailable_teacher_never_creates_a_training_client(self):
        self.service.supported_models.remove("Qwen/Qwen3.8-27B")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "unavailable"):
            await p.run_experiment(self.args(mode="opd"))
        self.assertEqual(self.service.training_clients, [])
        self.assertEqual(self.results()[0]["status"], "failed")

    async def test_sft_solution_dataset_is_checked_and_its_hash_recorded(self):
        path = self.data / "sft-solutions.jsonl"
        targets = [row | {"solution": r"A prepared solution ends in \boxed{70}."} for row in rows(80)]
        p.write_rows(path, targets)
        (self.root / "experiment.py").write_text("EXPERIMENT = " + repr(config(mode="sft", steps=2,
                                                                             sft_data="data/sft-solutions.jsonl").__dict__))
        with redirect_stdout(io.StringIO()):
            await p.run_experiment(self.args())
        result = self.results()[0]
        self.assertEqual(result["sft_data_sha256"], p.sha256(path))
        self.assertEqual(result["training"]["optimizer_steps"], 2)
        self.assertEqual(set(self.service.training_clients[0].loss_fns), {"cross_entropy"})
        bad = targets[:]
        bad[0] = bad[0] | {"problem": rows(30, "aime")[0]["problem"]}
        p.write_rows(path, bad)
        with self.assertRaisesRegex(ValueError, "overlap"):
            p.sft_training_rows(config(mode="sft", sft_data=str(path)), rows(80), rows(30, "aime"))

    async def test_fresh_candidates_and_replay_without_retraining(self):
        with redirect_stdout(io.StringIO()):
            await p.run_experiment(self.args(name="candidate-a"))
            first = self.results()[0]
            await p.run_experiment(self.args(command="evaluate", name="confirm-a", eval_seed=1042,
                                             run=self.root / "runs" / first["run_id"]))
            self.assertEqual(len(self.service.training_clients), 1)
            await p.run_experiment(self.args(name="candidate-b"))
        self.assertEqual(len(self.service.training_clients), 2)
        for result in self.results():
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["metrics"]["eval/generations"], 90)
            self.assertTrue(result["model_path"].startswith("trio://fake/"))
            self.assertEqual(result["prepare_sha256"], p.sha256(Path(p.__file__)))
            self.assertEqual(result["evaluation_sha256"], p.sha256(Path(p.evaluation.__file__)))

    def test_standalone_evaluation_cli_uses_base_or_saved_run_without_training(self):
        with redirect_stdout(io.StringIO()):
            p.evaluation.main(["--name", "standalone-base", "--eval-thinking", "--swanlab-mode", "disabled"])
            original = self.results()[0]
            self.assertEqual(original["experiment"]["mode"], "prompt")
            self.assertIsNone(original["model_path"])
            self.assertTrue(original["protocol"]["enable_thinking"])
            # Re-evaluation uses the saved instruction even after the current file changes.
            changed = original["experiment"] | {"prompt": "A different unevaluated instruction."}
            (self.root / "experiment.py").write_text("EXPERIMENT = " + repr(changed))
            p.evaluation.main(["--run", str(self.root / "runs" / original["run_id"]),
                               "--name", "standalone-confirm", "--eval-seed", "1042",
                               "--swanlab-mode", "disabled"])
        confirmation = next(row for row in self.results() if row["name"] == "standalone-confirm")
        self.assertEqual(confirmation["kind"], "reevaluation")
        self.assertEqual(confirmation["experiment"]["prompt"], original["experiment"]["prompt"])
        self.assertTrue(confirmation["protocol"]["enable_thinking"])
        self.assertEqual(confirmation["protocol"]["seed"], 1042)
        self.assertEqual(confirmation["metrics"]["eval/generations"], 90)
        self.assertEqual(self.service.training_clients, [])
        self.assertEqual(len(self.service.sampler.calls), 60)

    async def test_failure_is_saved_without_a_full_score(self):
        self.service.sampler.fail_seed = p.EVAL_SEED + 9
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            await p.run_experiment(self.args(mode="prompt"))
        result = self.results()[0]
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("metrics", result)
        self.assertEqual(len(self.service.training_clients), 0)

    async def test_real_swanlab_offline_recording_with_fake_model_service(self):
        with redirect_stdout(io.StringIO()):
            await p.run_experiment(self.args(mode="prompt", swanlab_mode="offline"))
        result = self.results()[0]
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["swanlab"]["id"])
        self.assertIsNone(result["swanlab"]["url"])
        self.assertEqual(len(self.service.training_clients), 0)

    def test_dataset_tampering_is_rejected_before_service_creation(self):
        with (self.data / "aime25.jsonl").open("a") as output:
            output.write("{}\n")
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            p.load_data()
        self.service_patch.target.ServiceClient.assert_not_called()


if __name__ == "__main__":
    unittest.main()
