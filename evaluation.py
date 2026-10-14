"""Fixed AIME25 grading, sampling protocol, and concurrent evaluation.

Run without --run to evaluate the current prompt on the 4B base model.
Pass --run runs/RUN_ID to re-evaluate that run's saved model and prompt.
Importing this module never contacts the model service.
"""

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import pytrio as trio


EVAL_SOURCE = "math-ai/aime25"
EVAL_REVISION = "563bb8404243c5f09de6ec262f2db674fe5bce9b"
EVAL_FILE_SHA256 = "b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331"
MAX_TOKENS = 1024
MAX_PROMPT_TOKENS = 2048
EVAL_N = 3
EVAL_ROWS = 30
EVAL_CONCURRENCY = 16
EVAL_TEMPERATURE = 0.8
EVAL_SEED = 42
PROTOCOL = {
    "dataset": EVAL_SOURCE,
    "revision": EVAL_REVISION,
    "rows": EVAL_ROWS,
    "n": EVAL_N,
    "max_tokens": MAX_TOKENS,
    "temperature": EVAL_TEMPERATURE,
    "top_p": 1.0,
    "seed": EVAL_SEED,
    "enable_thinking": False,
    "tools": False,
    "concurrency": EVAL_CONCURRENCY,
}


def integer_answer(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip().replace("$", "").replace(",", "")
    value = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", value)
    if len(value) > 1000 or not re.fullmatch(r"\s*[+-]?\d+\s*", value):
        return None
    return str(int(value))


def last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    return None


def grade(text: str, answer: str) -> tuple[bool, bool]:
    prediction = integer_answer(last_boxed(text))
    return prediction is not None and prediction == answer, prediction is not None


def validate_aime(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EVAL_ROWS or len({row["id"] for row in rows}) != EVAL_ROWS:
        raise ValueError("Evaluation requires exactly 30 distinct AIME25 questions.")
    if any(integer_answer(row["answer"]) != row["answer"] or
           not 0 <= int(row["answer"]) <= 999 or not row["problem"].strip() for row in rows):
        raise ValueError("Invalid AIME25 problem or integer answer.")


def prompt_ids(tokenizer: Any, problem: str, instruction: str, *, enable_thinking: bool = False) -> list[int]:
    if type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean.")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\n{instruction}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    if not tokens or len(tokens) > MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt length {len(tokens)} is outside [1, {MAX_PROMPT_TOKENS}].")
    return tokens


async def gather_complete(tasks):
    tasks = [asyncio.create_task(task) for task in tasks]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def sample_group(sampler, tokenizer, row, config, n, temperature, seed, semaphore, *, enable_thinking=False):
    tokens = prompt_ids(tokenizer, row["problem"], config.prompt, enable_thinking=enable_thinking)
    params = trio.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature,
                                top_p=1.0, seed=seed,
                                stop=[tokenizer.eos_token] if tokenizer.eos_token else None)
    async with semaphore:
        response = await sampler.sample_async(prompt=trio.ModelInput.from_ints(tokens),
                                              num_samples=n, sampling_params=params, return_text=True)
    if len(response.sequences) != n or response.input_tokens != len(tokens):
        raise ValueError("Sampler returned an incomplete group or mismatched prompt tokens.")
    samples = []
    for sequence in response.sequences:
        completion = list(sequence.tokens)
        if not completion or len(completion) > MAX_TOKENS:
            raise ValueError("Sampler returned an empty or over-budget completion.")
        text = sequence.text if sequence.text is not None else tokenizer.decode(completion)
        correct, formatted = grade(text, row["answer"])
        samples.append({"prompt_tokens": tokens, "tokens": completion,
                        "logprobs": list(sequence.logprobs), "text": text,
                        "prediction": integer_answer(last_boxed(text)),
                        "correct": correct, "formatted": formatted,
                        "stop_reason": sequence.stop_reason,
                        "hit_max_tokens": len(completion) == MAX_TOKENS})
    return samples


async def evaluate(sampler, tokenizer, rows, config, output: Path, seed=EVAL_SEED, *, enable_thinking=False):
    validate_aime(rows)
    started = time.monotonic()
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)

    async def one(index, row):
        samples = await sample_group(sampler, tokenizer, row, config, EVAL_N,
                                     EVAL_TEMPERATURE, seed + index, semaphore,
                                     enable_thinking=enable_thinking)
        generations = [{key: sample[key] for key in
                        ("text", "prediction", "correct", "formatted", "stop_reason", "hit_max_tokens")}
                       | {"completion_tokens": len(sample["tokens"])} for sample in samples]
        result = {"id": row["id"], "answer": row["answer"],
                  "passed": any(sample["correct"] for sample in samples), "generations": generations}
        # Persist each completed question, even if a later request fails.
        with output.open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"AIME25 {index + 1:02d}/30 complete", flush=True)
        return result

    results = await gather_complete(one(index, row) for index, row in enumerate(rows))
    generations = [g for result in results for g in result["generations"]]
    return {"eval/pass_at_3": sum(result["passed"] for result in results) / EVAL_ROWS,
            "eval/average_at_3": sum(g["correct"] for g in generations) / len(generations),
            "eval/format_rate": sum(g["formatted"] for g in generations) / len(generations),
            "eval/hit_max_tokens_rate": sum(g["hit_max_tokens"] for g in generations) / len(generations),
            "eval/completion_tokens": sum(g["completion_tokens"] for g in generations),
            "eval/seconds": time.monotonic() - started,
            "eval/problems": EVAL_ROWS, "eval/generations": len(generations)}

def add_cli_arguments(parser):
    """Shared evaluation and tracking options for the training and eval CLIs."""
    parser.add_argument("--name", required=True)
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "local", "disabled"], default="offline")
    parser.add_argument("--swanlab-project", default="autoresearch-democase")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help="Keep fixed for comparable trials; change for paired confirmation.")
    parser.add_argument("--eval-thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable evaluation thinking. Default: off for a base run; inherited for re-evaluation.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_arguments(parser)
    parser.add_argument("--run", type=Path, help="Existing run directory. Omit to evaluate the current base-model prompt.")
    args = parser.parse_args(argv)
    args.command = "evaluate" if args.run is not None else "run"
    args.mode, args.base_model, args.teacher_base_model = "prompt", None, None
    # Reuse artifact tracking and service setup; this entry point never trains.
    from prepare import run_experiment
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
