"""Fixed data preparation, training budgets, run orchestration, and logging.

Nothing runs at import time. Preparing/checking data needs no model service;
only the explicit `run` and `evaluate` commands generate model responses.
"""

import argparse
import ast
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from typing import Any, Iterable
import urllib.request
import uuid

import numpy as np
import pytrio as trio

import evaluation
from evaluation import (
    EVAL_CONCURRENCY, EVAL_FILE_SHA256, EVAL_N, EVAL_REVISION, EVAL_ROWS,
    EVAL_SEED, EVAL_SOURCE, EVAL_TEMPERATURE, MAX_PROMPT_TOKENS, MAX_TOKENS,
    PROTOCOL, add_cli_arguments, evaluate, gather_complete, grade, integer_answer,
    last_boxed, prompt_ids, sample_group, validate_aime,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
TARGET_MODEL = "Qwen/Qwen3.5-4B"
TRAIN_SOURCE = "BytedTsinghua-SIA/DAPO-Math-17k"
TRAIN_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
TRAIN_FILE_SHA256 = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
MAX_STEPS = 20
TRAIN_BATCH_SIZE = 4
TRAIN_GROUP_SIZE = 4
MICRO_BATCH_PADDED_TOKENS = 16384
TRAIN_SEED = 42
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


@dataclass(frozen=True)
class Experiment:
    mode: str
    hypothesis: str
    prompt: str
    steps: int
    learning_rate: float
    lora_rank: int
    train_temperature: float
    base_model: str = TARGET_MODEL
    teacher_base_model: str | None = None
    teacher_model_path: str | None = None
    group_size: int = TRAIN_GROUP_SIZE
    opd_kl_coef: float = 1.0
    sft_data: str | None = None

    def validate(self) -> None:
        if self.mode not in {"prompt", "grpo", "sft", "opd"}:
            raise ValueError("mode must be prompt, grpo, sft, or opd.")
        if type(self.steps) is not int or not 0 <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be an integer in [0, {MAX_STEPS}].")
        if self.mode != "prompt" and self.steps == 0:
            raise ValueError("Training requires at least one step.")
        if self.base_model != TARGET_MODEL:
            raise ValueError(f"The optimization target is fixed to {TARGET_MODEL}; choose larger models as OPD teachers.")
        if self.mode == "opd" and (not isinstance(self.teacher_base_model, str) or not self.teacher_base_model.strip()):
            raise ValueError("OPD requires teacher_base_model.")
        if self.teacher_model_path is not None and not self.teacher_model_path.startswith("trio://"):
            raise ValueError("teacher_model_path must be a trio:// sampler checkpoint.")
        if type(self.group_size) is not int or not 1 <= self.group_size <= TRAIN_GROUP_SIZE:
            raise ValueError(f"group_size must be an integer in [1, {TRAIN_GROUP_SIZE}].")
        if self.mode == "grpo" and self.group_size < 2:
            raise ValueError("GRPO requires at least two samples per problem.")
        if not math.isfinite(self.opd_kl_coef) or self.opd_kl_coef <= 0:
            raise ValueError("opd_kl_coef must be finite and positive.")
        if self.sft_data is not None and not isinstance(self.sft_data, str):
            raise ValueError("sft_data must be a local JSONL path or None.")
        if type(self.lora_rank) is not int or not 1 <= self.lora_rank <= 64:
            raise ValueError("lora_rank must be an integer in [1, 64].")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1e-3:
            raise ValueError("learning_rate must be finite and in (0, 1e-3].")
        if not math.isfinite(self.train_temperature) or not 0 < self.train_temperature <= 2:
            raise ValueError("train_temperature must be finite and in (0, 2].")
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 4000:
            raise ValueError("prompt must be a nonempty instruction of at most 4000 characters.")
        if not isinstance(self.hypothesis, str) or not self.hypothesis.strip():
            raise ValueError("Describe the experimental hypothesis.")


def read_experiment(path: Path = ROOT / "experiment.py", mode: str | None = None,
                    base_model: str | None = None, teacher_base_model: str | None = None) -> Experiment:
    # Read fresh literal configuration without executing model-service code.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "EXPERIMENT"]
    if len(assignments) != 1:
        raise ValueError("Define exactly one literal EXPERIMENT dictionary.")
    values = dict(ast.literal_eval(assignments[0].value))
    for key, value in {"mode": mode, "base_model": base_model, "teacher_base_model": teacher_base_model}.items():
        if value is not None:
            values[key] = value
    config = Experiment(**values)
    config.validate()
    return config


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_question(problem: str) -> str:
    return problem.removeprefix(DAPO_PREFIX).removesuffix(DAPO_SUFFIX).strip()


def question_key(problem: str) -> str:
    return re.sub(r"\s+", "", clean_question(problem))


def collect_train(raw_rows: Iterable[dict[str, Any]], aime: list[dict[str, Any]], size: int):
    excluded = {question_key(row["problem"]) for row in aime}
    records: dict[str, dict[str, str]] = {}
    conflicts: set[str] = set()
    cached_questions: dict[str, tuple[str, str]] = {}
    cached_answers: dict[str, str | None] = {}
    counts = {"scanned": 0, "duplicates": 0, "eval_overlap": 0, "invalid": 0,
              "conflicting_questions": 0}
    for row in raw_rows:
        counts["scanned"] += 1
        if counts["scanned"] % 100000 == 0:
            print(f"Scanned {counts['scanned']:,} raw DAPO rows", flush=True)
        prompt = row.get("prompt")
        reward = row.get("reward_model") or {}
        if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], dict) or not isinstance(reward, dict):
            counts["invalid"] += 1
            continue
        answer = reward.get("ground_truth")
        if isinstance(answer, list):
            answer = answer[0] if len(answer) == 1 else None
        answer_key = str(answer)
        if answer_key not in cached_answers:
            cached_answers[answer_key] = integer_answer(answer)
        answer = cached_answers[answer_key]
        raw_question = str(prompt[0].get("content", ""))
        if raw_question not in cached_questions:
            problem = clean_question(raw_question)
            cached_questions[raw_question] = (problem, question_key(problem))
        problem, key = cached_questions[raw_question]
        if not problem or answer is None:
            counts["invalid"] += 1
            continue
        if key in excluded:
            counts["eval_overlap"] += 1
            continue
        if key in conflicts:
            counts["duplicates"] += 1
            continue
        if key in records:
            if records[key]["answer"] != answer:
                conflicts.add(key)
                del records[key]
                counts["conflicting_questions"] += 1
            counts["duplicates"] += 1
            continue
        records[key] = {"id": hashlib.sha256(key.encode()).hexdigest()[:16],
                        "problem": problem, "answer": answer}
    selected = list(records.values())
    counts["unique_usable"] = len(selected)
    if len(selected) < size:
        raise ValueError(f"Only {len(selected)} unique consistent integer-answer questions found; need {size}.")
    random.Random(TRAIN_SEED).shuffle(selected)
    return selected[:size], counts


def cache_source(path: Path, url: str, expected_sha256: str, local_source: Path | None = None) -> None:
    if path.exists():
        if sha256(path) != expected_sha256:
            raise ValueError(f"Raw dataset checksum mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.part")
    try:
        if local_source is not None:
            print(f"Copying cached raw dataset: {path.name}", flush=True)
            shutil.copyfile(local_source, temporary)
        else:
            print(f"Downloading full raw dataset: {path.name}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256(temporary) != expected_sha256:
            raise ValueError(f"Source dataset checksum mismatch: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_data(size: int, parquet: Path | None) -> None:
    import pyarrow.parquet as pq

    if size < TRAIN_BATCH_SIZE * MAX_STEPS:
        raise ValueError(f"Prepare at least {TRAIN_BATCH_SIZE * MAX_STEPS} unique questions.")
    raw_directory = DATA / "raw"
    sources = {
        "aime25.jsonl": {"dataset": EVAL_SOURCE, "revision": EVAL_REVISION,
                         "url": f"https://huggingface.co/datasets/{EVAL_SOURCE}/resolve/{EVAL_REVISION}/test.jsonl",
                         "sha256": EVAL_FILE_SHA256},
        "dapo-math-17k.parquet": {"dataset": TRAIN_SOURCE, "revision": TRAIN_REVISION,
                                  "url": f"https://huggingface.co/datasets/{TRAIN_SOURCE}/resolve/{TRAIN_REVISION}/data/dapo-math-17k.parquet",
                                  "sha256": TRAIN_FILE_SHA256},
    }
    for name, source in sources.items():
        cache_source(raw_directory / name, source["url"], source["sha256"],
                     local_source=parquet if name.endswith(".parquet") else None)
        source["bytes"] = (raw_directory / name).stat().st_size
    write_json(raw_directory / "sources.json", sources)
    aime = read_rows(raw_directory / "aime25.jsonl")
    aime = [{"id": str(row["id"]), "problem": str(row["problem"]).strip(),
             "answer": integer_answer(row["answer"])} for row in aime]
    validate_aime(aime)
    provenance = {"dataset": TRAIN_SOURCE, "revision": TRAIN_REVISION,
                  "local_file": "raw/dapo-math-17k.parquet", "sha256": TRAIN_FILE_SHA256}
    batches = pq.ParquetFile(raw_directory / "dapo-math-17k.parquet").iter_batches(
        batch_size=8192, columns=["prompt", "reward_model"]
    )
    raw = (row for batch in batches for row in batch.to_pylist())
    train, counts = collect_train(raw, aime, size)
    write_rows(DATA / "train.jsonl", train)
    write_rows(DATA / "aime25.jsonl", aime)
    manifest = {"train_source": provenance, "eval_source": EVAL_SOURCE,
                "eval_revision": EVAL_REVISION, "train_rows": len(train),
                "eval_rows": len(aime), "integer_answers_only": True,
                "selection_seed": TRAIN_SEED, "cleaning": counts,
                "sha256": {name: sha256(DATA / name) for name in ("train.jsonl", "aime25.jsonl")}}
    write_json(DATA / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def load_data():
    if not (DATA / "manifest.json").exists():
        raise ValueError("Prepare the data first: uv run python prepare.py prepare")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["eval_source"] != EVAL_SOURCE or manifest["eval_revision"] != EVAL_REVISION:
        raise ValueError("Evaluation dataset differs from the pinned AIME25 version.")
    for name in ("train.jsonl", "aime25.jsonl"):
        if sha256(DATA / name) != manifest["sha256"][name]:
            raise ValueError(f"Prepared dataset changed: {name}")
    train, aime = read_rows(DATA / "train.jsonl"), read_rows(DATA / "aime25.jsonl")
    validate_aime(aime)
    if len(train) != manifest["train_rows"] or len(train) < TRAIN_BATCH_SIZE * MAX_STEPS:
        raise ValueError("Invalid training dataset size.")
    keys = [question_key(row["problem"]) for row in train]
    if len(keys) != len(set(keys)) or set(keys) & {question_key(row["problem"]) for row in aime}:
        raise ValueError("Training data contains duplicates or evaluation overlap.")
    if any(not row["problem"].strip() or integer_answer(row["answer"]) != row["answer"] for row in train):
        raise ValueError("Invalid training example.")
    return train, aime, manifest


def pack_samples(samples):
    batches, current = [], []
    longest = 0
    for sample in sorted(samples, key=lambda s: len(s["prompt_tokens"]) + len(s["tokens"]), reverse=True):
        length = len(sample["prompt_tokens"]) + len(sample["tokens"]) - 1
        if current and (len(current) + 1) * max(longest, length) > MICRO_BATCH_PADDED_TOKENS:
            batches.append(current)
            current, longest = [], 0
        current.append(sample)
        longest = max(longest, length)
    return batches + ([current] if current else [])


def grpo_datum(sample, scale=1.0):
    """Right-shifted importance-sampling input for GRPO or token-level OPD."""
    tokens, old_logprobs = sample["tokens"], sample["logprobs"]
    if len(tokens) != len(old_logprobs) or any(value is None or not math.isfinite(value) for value in old_logprobs):
        raise ValueError("Completion tokens and rollout logprobs must align and be finite.")
    prompt = sample["prompt_tokens"]
    padding = len(prompt) - 1
    advantages = sample.get("advantages")
    if advantages is None:
        advantages = [sample["advantage"]] * len(tokens)
    if len(advantages) != len(tokens) or any(not math.isfinite(value) for value in advantages):
        raise ValueError("Policy advantages must align with completion tokens and be finite.")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(prompt + tokens[:-1]),
        loss_fn_inputs={"target_tokens": np.asarray([0] * padding + tokens, dtype=np.int64),
                        "logprobs": np.asarray([0.0] * padding + old_logprobs, dtype=np.float32),
                        "advantages": np.asarray([0.0] * padding +
                                                 [value * scale for value in advantages], dtype=np.float32)},
    )


def sft_sample(tokenizer, row, config):
    prompt = prompt_ids(tokenizer, row["problem"], config.prompt)
    solution = row.get("solution") or ("\\boxed{" + row["answer"] + "}")
    if not grade(solution, row["answer"])[0]:
        raise ValueError("SFT solution's final boxed answer does not match its training label.")
    tokens = tokenizer.encode(solution, add_special_tokens=False)
    if tokenizer.eos_token:
        eos = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        if eos and tokens[-len(eos):] != eos:
            tokens += eos
    if not tokens or len(tokens) > MAX_TOKENS:
        raise ValueError("SFT target must fit the 1024-token budget, including EOS.")
    return {"prompt_tokens": prompt, "tokens": tokens, "correct": True,
            "formatted": True, "hit_max_tokens": len(tokens) == MAX_TOKENS,
            "answer_only": not bool(row.get("solution"))}


def sft_datum(sample):
    prompt, tokens = sample["prompt_tokens"], sample["tokens"]
    padding = len(prompt) - 1
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(prompt + tokens[:-1]),
        loss_fn_inputs={"target_tokens": np.asarray([0] * padding + tokens, dtype=np.int64),
                        "weights": np.asarray([0.0] * padding + [1.0] * len(tokens), dtype=np.float32)},
    )


def sft_training_rows(config, train, aime):
    if config.sft_data is None:
        return train, None
    path = Path(config.sft_data)
    if not path.is_absolute():
        path = ROOT / path
    records = read_rows(path)
    keys = set()
    excluded = {question_key(row["problem"]) for row in aime}
    for index, row in enumerate(records):
        row["problem"] = clean_question(str(row["problem"]))
        row["answer"] = integer_answer(row["answer"])
        row["id"] = str(row.get("id", index))
        key = question_key(row["problem"])
        if not key or row["answer"] is None or key in keys or key in excluded:
            raise ValueError("SFT data contains invalid labels, duplicates, or AIME25 overlap.")
        if row.get("solution") and not grade(row["solution"], row["answer"])[0]:
            raise ValueError("SFT solution's final boxed answer disagrees with its label.")
        keys.add(key)
    if len(records) < TRAIN_BATCH_SIZE * config.steps:
        raise ValueError("SFT data does not contain enough unique examples for the requested steps.")
    random.Random(TRAIN_SEED).shuffle(records)
    return records, sha256(path)


def check_opd_tokenizers(student_tokenizer, teacher_tokenizer):
    # No model-name allowlist. Validate the token IDs used for this model pair.
    student_vocab, teacher_vocab = student_tokenizer.get_vocab(), teacher_tokenizer.get_vocab()
    if any(teacher_vocab.get(token) != token_id for token, token_id in student_vocab.items()):
        raise ValueError("OPD teacher/student token vocabularies differ; choose a token-compatible pair.")


async def opd_advantages(teacher, sample, coef, semaphore):
    ids = sample["prompt_tokens"] + sample["tokens"]
    async with semaphore:
        all_logprobs = await teacher.compute_logprobs_async(trio.ModelInput.from_ints(ids))
    teacher_logprobs = all_logprobs[len(sample["prompt_tokens"]):]
    if len(all_logprobs) != len(ids) or len(teacher_logprobs) != len(sample["tokens"]):
        raise ValueError("Teacher logprobs do not align with the student trajectory.")
    if any(value is None or not math.isfinite(value) for value in teacher_logprobs):
        raise ValueError("Teacher completion logprobs must be finite.")
    if len(sample["logprobs"]) != len(sample["tokens"]) or any(
            value is None or not math.isfinite(value) for value in sample["logprobs"]):
        raise ValueError("Student rollout logprobs must align and be finite.")
    reverse_kl = [old - target for old, target in zip(sample["logprobs"], teacher_logprobs, strict=True)]
    sample["advantages"] = [-coef * value for value in reverse_kl]
    return reverse_kl


async def train_model(training, tokenizer, rows, config, log, teacher=None):
    config.validate()
    if config.mode == "prompt":
        raise ValueError("Prompt-only experiments do not train weights.")
    if config.mode == "opd":
        if teacher is None:
            raise ValueError("OPD requires a frozen teacher client.")
        check_opd_tokenizers(tokenizer, teacher.get_tokenizer())
    updates, empty_rounds = 0, 0
    for round_index in range(config.steps):
        started = time.monotonic()
        batch = rows[round_index * TRAIN_BATCH_SIZE:(round_index + 1) * TRAIN_BATCH_SIZE]
        if len(batch) != TRAIN_BATCH_SIZE:
            raise ValueError("Insufficient unique training examples for this round.")
        metrics = {}
        degenerate_groups = 0
        if config.mode == "sft":
            groups = [[sft_sample(tokenizer, row, config)] for row in batch]
            usable = [group[0] for group in groups]
            metrics["sft/answer_only_fraction"] = sum(s["answer_only"] for s in usable) / len(usable)
            metrics["sft/supervised_tokens"] = sum(len(s["tokens"]) for s in usable)
        else:
            sampler = await training.save_weights_and_get_sampling_client_async()
            semaphore = asyncio.Semaphore(TRAIN_BATCH_SIZE)
            groups = await gather_complete(
                sample_group(sampler, tokenizer, row, config, config.group_size,
                             config.train_temperature, TRAIN_SEED + round_index * TRAIN_BATCH_SIZE + i,
                             semaphore) for i, row in enumerate(batch)
            )
            if config.mode == "opd":
                teacher_semaphore = asyncio.Semaphore(TRAIN_BATCH_SIZE)
                kls = await gather_complete(
                    opd_advantages(teacher, sample, config.opd_kl_coef, teacher_semaphore)
                    for group in groups for sample in group
                )
                kls = [value for sample_kls in kls for value in sample_kls]
                metrics.update({"opd/reverse_kl_mean": float(np.mean(kls)),
                                "opd/reverse_kl_std": float(np.std(kls))})
            usable = []
            for group in groups:
                if config.mode == "grpo":
                    mean_reward = sum(s["correct"] for s in group) / len(group)
                    for sample in group:
                        sample["advantages"] = [float(sample["correct"]) - mean_reward] * len(sample["tokens"])
                if any(any(s["advantages"]) for s in group):
                    usable.extend(group)
                else:
                    degenerate_groups += 1
        make_datum = sft_datum if config.mode == "sft" else grpo_datum
        # Validate the entire logical batch before submitting any backward call.
        for sample in usable:
            make_datum(sample)
        for micro_batch in pack_samples(usable):
            scale = len(micro_batch) / len(usable)
            datums = [sft_datum(sample) if config.mode == "sft" else grpo_datum(sample, scale)
                      for sample in micro_batch]
            future = await training.forward_backward_async(
                datums, loss_fn="cross_entropy" if config.mode == "sft" else "importance_sampling"
            )
            result = await future
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[f"trainer/{key}"] = metrics.get(f"trainer/{key}", 0.0) + scale * value
        if usable:
            if updates >= MAX_STEPS:
                raise RuntimeError("Optimizer update budget exhausted.")
            future = await training.optim_step_async(trio.AdamParams(learning_rate=config.learning_rate))
            await future
            updates += 1
            empty_rounds = 0
        else:
            empty_rounds += 1
        all_samples = [sample for group in groups for sample in group]
        metrics.update({"train/degenerate_fraction": degenerate_groups / len(groups),
             "train/optimizer_steps": updates, "train/update_skipped": int(not usable),
             "train/completion_tokens": sum(len(s["tokens"]) for s in all_samples),
             "train/hit_max_tokens_rate": sum(s["hit_max_tokens"] for s in all_samples) / len(all_samples),
             "train/step_seconds": time.monotonic() - started})
        if config.mode != "sft":
            metrics["train/reward"] = sum(s["correct"] for s in all_samples) / len(all_samples)
        log(metrics, round_index + 1)
        if empty_rounds == 3:
            break
    return {"optimizer_steps": updates, "training_rounds": round_index + 1,
            "rollout_rounds": 0 if config.mode == "sft" else round_index + 1,
            "early_stop_no_signal": empty_rounds == 3}


def validate_supported_models(service, config, require_teacher=False):
    supported = service.get_supported_models()
    requested = [config.base_model]
    if require_teacher:
        requested.append(config.teacher_base_model)
    missing = [model for model in requested if model not in supported]
    if missing:
        raise ValueError(f"Models unavailable in the current PyTRIO catalog: {missing}. Available: {supported}")
    return supported


async def run_experiment(args):
    from dataclasses import asdict
    import swanlab

    train, aime, manifest = load_data()
    source_run = None
    if args.command == "evaluate":
        source_run = json.loads((args.run / "result.json").read_text())
        if source_run["status"] != "complete" or source_run["data_sha256"] != manifest["sha256"]:
            raise ValueError("Re-evaluation requires a completed run with unchanged datasets.")
        config = Experiment(**source_run["experiment"])
        config.validate()
        base_model, model_path = source_run["base_model"], source_run["model_path"]
        if base_model != config.base_model:
            raise ValueError("Saved evaluation model does not match the fixed 4B target.")
    else:
        config = read_experiment(mode=args.mode, base_model=args.base_model,
                                 teacher_base_model=args.teacher_base_model)
        base_model, model_path = config.base_model, None
    eval_thinking = args.eval_thinking
    if eval_thinking is None:
        eval_thinking = (source_run["protocol"] if source_run else PROTOCOL)["enable_thinking"]
    if type(eval_thinking) is not bool:
        raise ValueError("Evaluation thinking setting must be a boolean.")
    sft_data_sha256 = source_run.get("sft_data_sha256") if source_run else None
    if source_run is None and config.mode == "sft":
        train, sft_data_sha256 = sft_training_rows(config, train, aime)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.name):
        raise ValueError("Use a short experiment name containing letters, numbers, '.', '_' or '-'.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.name + "-" + uuid.uuid4().hex[:6]
    directory = RUNS / run_id
    directory.mkdir(parents=True)
    snapshot = args.run / "experiment.py" if source_run else ROOT / "experiment.py"
    shutil.copyfile(snapshot, directory / "experiment.py")
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {"run_id": run_id, "name": args.name, "status": "running",
              "kind": "reevaluation" if source_run else "experiment",
              "source_run": source_run["run_id"] if source_run else None,
              "base_model": base_model, "model_path": model_path,
              "experiment": asdict(config), "code_commit": code_commit,
              "experiment_sha256": sha256(directory / "experiment.py"),
              "prepare_sha256": sha256(Path(__file__)),
              "evaluation_sha256": sha256(Path(evaluation.__file__)), "data_sha256": manifest["sha256"],
              "sft_data_sha256": sft_data_sha256,
              "protocol": PROTOCOL | {"seed": args.eval_seed, "enable_thinking": eval_thinking},
              "swanlab": {"mode": args.swanlab_mode}}
    write_json(directory / "result.json", result)
    tracking = None

    def log(metrics, step):
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        if "train/optimizer_steps" in metrics:
            result["training"] = {"optimizer_steps": metrics["train/optimizer_steps"],
                                  "training_rounds": step,
                                  "rollout_rounds": 0 if config.mode == "sft" else step}
            write_json(directory / "result.json", result)
        if tracking is not None:
            tracking.log(metrics, step=step)

    try:
        service = trio.ServiceClient()
        result["supported_models"] = validate_supported_models(
            service, config, require_teacher=source_run is None and config.mode == "opd"
        )
        if args.swanlab_mode != "disabled":
            tracking = swanlab.init(project=args.swanlab_project, workspace=args.swanlab_workspace,
                                    name=run_id, mode=args.swanlab_mode, public=False,
                                    log_dir=str(ROOT / "swanlog"), config=result,
                                    description=config.hypothesis)
            result["swanlab"].update({"id": tracking.id,
                                      "url": tracking.url if args.swanlab_mode == "online" else None})
            write_json(directory / "result.json", result)
        train_summary = {"optimizer_steps": 0, "training_rounds": 0,
                         "rollout_rounds": 0, "early_stop_no_signal": False}
        if source_run is None and config.mode != "prompt":
            training = await service.create_lora_training_client_async(
                base_model=base_model, rank=config.lora_rank, seed=TRAIN_SEED
            )
            tokenizer = training.get_tokenizer()
            teacher = None
            if config.mode == "opd":
                teacher = await service.create_sampling_client_async(
                    base_model=config.teacher_base_model, model_path=config.teacher_model_path
                )
            train_summary = await train_model(training, tokenizer, train, config, log, teacher=teacher)
            future = await training.save_weights_for_sampler_async(name=run_id)
            model_path = (await future).path
            result.update({"model_path": model_path, "training": train_summary})
            write_json(directory / "result.json", result)
        if tracking is not None:
            tracking.config.update({"model_path": model_path, "training": train_summary})
        sampler = await service.create_sampling_client_async(base_model=base_model, model_path=model_path)
        metrics = await evaluate(sampler, sampler.get_tokenizer(), aime, config,
                                 directory / "generations.jsonl", seed=args.eval_seed,
                                 enable_thinking=eval_thinking)
        log(metrics, train_summary["training_rounds"] + 1)
        result.update({"status": "complete", "training": train_summary, "metrics": metrics})
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        write_json(directory / "result.json", result)
        with (ROOT / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        if tracking is not None:
            tracking.finish(state="success" if result["status"] == "complete" else "crashed",
                            error=result.get("error"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare pinned datasets; no training or sampling.")
    prepare.add_argument("--train-size", type=int, default=1024)
    prepare.add_argument("--dapo-parquet", type=Path, help="Optional existing local parquet; its SHA256 is recorded.")
    check = commands.add_parser("check", help="Validate local configuration and data; no model service calls.")
    check.add_argument("--eval-thinking", action=argparse.BooleanOptionalAction,
                       default=PROTOCOL["enable_thinking"], help="Preview the evaluation thinking setting locally.")
    commands.add_parser("summary", help="Print local experiment results; no model service calls.")
    for command in ("run", "evaluate"):
        subparser = commands.add_parser(command, help="Explicitly run a model experiment or re-evaluate saved weights.")
        add_cli_arguments(subparser)
        if command == "run":
            subparser.add_argument("--base-model", choices=[TARGET_MODEL], default=None, help="Fixed 4B optimization target.")
            subparser.add_argument("--teacher-base-model", default=None, help="Override the OPD teacher from experiment.py.")
            subparser.add_argument("--mode", choices=["prompt", "grpo", "sft", "opd"], default=None)
        else:
            subparser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(args.train_size, args.dapo_parquet)
    elif args.command == "check":
        from dataclasses import asdict
        from importlib.metadata import version
        config = read_experiment()
        train, aime, manifest = load_data()
        print(json.dumps({"experiment": asdict(config),
                          "protocol": PROTOCOL | {"enable_thinking": args.eval_thinking},
                          "train_rows": len(train), "eval_rows": len(aime),
                          "train_source": manifest["train_source"],
                          "versions": {name: version(name) for name in ("pytrio", "swanlab", "datasets")}}, indent=2))
    elif args.command == "summary":
        print("run_id\tkind\tstatus\teval_thinking\tpass@3\taverage@3\toptimizer_steps\tmodel_path")
        for path in sorted(RUNS.glob("*/result.json")):
            row = json.loads(path.read_text())
            metrics = row.get("metrics", {})
            print("\t".join(str(value) for value in (row["run_id"], row["kind"], row["status"],
                  row["protocol"]["enable_thinking"],
                  metrics.get("eval/pass_at_3", ""), metrics.get("eval/average_at_3", ""),
                  row.get("training", {}).get("optimizer_steps", ""), row.get("model_path"))))
    else:
        asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
