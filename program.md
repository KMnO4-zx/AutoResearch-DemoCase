# AutoResearch experiment instructions

You are the coding agent driving this project. This workflow is independent of
any particular agent product. Your job is to test a concrete hypothesis, observe
the result, keep useful changes, and use the evidence to choose the next trial.
The objective is to improve `Qwen/Qwen3.5-4B` AIME25 pass@3 relative to its initial
prompt baseline under the same evaluation settings. The student stays fixed at 4B.

## Autonomous budget

Run one initial prompt baseline and at most **10 candidate attempts**. A smaller
user budget takes precedence. Prompt-only optimization, GRPO, SFT, and OPD all
share this budget. Failed and discarded attempts count; a failure before a run
artifact is created must still be recorded. Do not restart the count after a
rollback or a new agent turn. Stop at the limit and report the results.

Accepted improvements must become local Git commits. Base the next edit on the
latest accepted commit; restore that version when a candidate fails or does not
improve. Reusing a code commit does not authorize resuming its trained weights:
each training candidate still starts from a fresh base-model LoRA.

For comparisons between coding agents, use a separate branch and worktree (or
clone) for each agent. Use the assigned branch/worktree when one is provided.
Keep each agent's run artifacts, decision ledger, and best.json in its own
directory; ignored files do not become isolated just by switching branches.
Each agent has its own 10-attempt budget and baseline. Prepared datasets may be
shared read-only, with identical hashes and evaluation settings across agents.

## Allowed changes and skills

- Edit `experiment.py`: prompt, hypothesis, `prompt`/`grpo`/`sft`/`opd` mode,
  teacher model/checkpoint, learning rate, LoRA rank, steps,
  training temperature, group size, OPD KL coefficient, and a prepared SFT dataset path.
- Prompt-only experiments are first-class candidates. Set `mode="prompt"`
  and change the instruction; they evaluate the 4B base model with zero weight
  updates and follow the same keep/commit/discard rules as training candidates.
- Keep `base_model` fixed to `Qwen/Qwen3.5-4B`. Any compatible model in the
  current PyTRIO catalog may serve as the OPD teacher, including larger models.
  The evaluation always measures the 4B student.
- OPD uses current student completions and teacher logprobs on those same
  tokens. Use a token-compatible teacher/student pair; the runner validates
  their vocabularies before rollout. Teacher generation is not the OPD target.
- SFT defaults to final-answer supervision. For reasoning SFT, select an
  existing local JSONL with verified final answers via `sft_data`. Keep that
  dataset fixed during the comparison; do not claim answer-only SFT teaches a
  full solution process.
- Keep `prepare.py`, `evaluation.py`, `plot_progress.py`, the dataset manifest, answer extraction, scoring, and the
  evaluation protocol fixed during the search. Report a fixed-utility bug and fix it
  as maintenance before starting a new comparable series.
- You may use **pytrio-skill** to inspect SDK contracts, diagnose service errors,
  and reason about training. Read the skill and relevant references first.
- You may use **swanlab-skill** to record, query, plot, and compare this project's
  experiments. Online logging is allowed when the account is authenticated;
  otherwise use offline logging. These operations are within the research task.
- Do not write benchmark answers, per-question lookup tables, or AIME examples
  into prompts or training data. Evaluate the saved candidate model and prompt.

Skill sources:

- [pytrio-skill](https://github.com/SwanHubX/pytrio-skill)
- [swanlab-skill](https://github.com/SwanHubX/SwanLab-Skill)

If the agent does not support automatic skill discovery, read the installed
`SKILL.md` and relevant references as ordinary files. The workflow still works.

## Fixed experiment contract

| Item | Budget or rule |
| --- | --- |
| Training source | Pinned DAPO-Math-17k, deduplicated integer-answer subset |
| Optimization target | `Qwen/Qwen3.5-4B` AIME25 pass@3 improvement over its initial baseline |
| Autonomous budget | One baseline + at most 10 candidate attempts, including failures |
| Candidate initialization | Training: fresh 4B LoRA; prompt-only: base model, zero updates |
| Code checkpoint | Commit each accepted improvement; discard restores latest accepted commit |
| Training budget | At most 20 training rounds and 20 `optim_step` calls |
| Training batch | 4 problems; GRPO 2–4 samples, OPD 1–4 samples, SFT one target per problem |
| Training/target output budget | At most 1024 tokens per completion/target, including EOS |
| Early stopping | Stop after 3 consecutive batches without relative reward signal |
| Evaluation | All 30 AIME25 problems, 3 samples per problem |
| Output budget | 1024 tokens per answer, including generated reasoning |
| Evaluation sampling | Temperature 0.8, top-p 1.0, seed 42 + problem index |
| Evaluation concurrency | At most 16 problem requests in flight |
| Evaluation thinking | Choose on/off before the baseline; fixed throughout a comparable series |
| Model interaction | Single-turn text; training thinking disabled; no tools |
| Primary score | Fraction of problems with at least one correct answer out of 3 |

Twenty updates limit training work; service queueing and generation length still
affect wall time. Prompt-only trials perform zero optimizer updates. Three
samples require stochastic decoding; do not change evaluation temperature to 0.
Use `--eval-thinking` or `--no-eval-thinking` to select the evaluation template.
New experiments default to thinking off; re-evaluation inherits the source run's
setting unless explicitly overridden. Generated reasoning counts toward the
same 1024-token output budget. Record and compare `protocol.enable_thinking`;
changing it requires a separate baseline and comparison series.

## Experiment loop

1. Read `experiment.py`, run `uv run python prepare.py check`, and check the
   existing results with `uv run python prepare.py summary`. Fix local problems
   before calling the model service. Inspect `git status --short` and use a
   dedicated research branch. Commit the reviewed project setup before the
   baseline if it is not yet tracked, staging exact project paths and excluding
   datasets/artifacts/credentials. Preserve unrelated work in an isolated
   worktree when necessary. The baseline must start from a clean code snapshot.
   On resuming a series, read `best.json` and its decisions, continue the existing
   attempt count, and verify HEAD is the selected commit before the next edit.
2. Keep the student at `Qwen/Qwen3.5-4B`. For OPD, select `teacher_base_model`
   and optionally a frozen teacher checkpoint. Use the user's evaluation thinking
   setting, defaulting to off when unspecified, for both baseline and candidates.
   The examples below use thinking off; replace the flag consistently if enabled.
3. If using online SwanLab, run `uv run swanlab ping` and `uv run swanlab verify`.
4. Start with one baseline using the initial prompt:

   ```bash
   uv run python evaluation.py --name baseline \
     --no-eval-thinking --swanlab-mode online
   ```

   Initialize `best.json` with this run's ID, score, model path, the baseline
   run ID, and full baseline commit SHA. Preserve the baseline artifact.

5. Write one concrete hypothesis in `experiment.py`, change a small number of
   related settings, and run a uniquely named candidate:

   ```bash
   uv run python prepare.py run --name concise-grpo \
     --no-eval-thinking --swanlab-mode online
   ```

   Select `prompt`, `grpo`, `sft`, or `opd` in the configuration or with `--mode`.
   OPD can use `group_size=1`; GRPO requires at least two samples. For example:

   ```bash
   uv run python prepare.py run --mode opd --name opd-trial \
     --no-eval-thinking --teacher-base-model TEACHER --swanlab-mode online
   ```
6. Read `runs/RUN_ID/result.json` and `metrics.jsonl`; use SwanLab to understand
   reward, OPD reverse-KL estimates, SFT supervised tokens, generation length, and evaluation results. A
   failed request makes the trial failed; partial evaluation is not a score.
7. Keep a candidate only when the 4B student's pass@3 strictly improves over the current best
   under the same evaluation protocol, including the thinking setting. Report
   improvement relative to the initial 4B baseline as well.
   A one-question gain is 3.33 percentage points and may be sampling noise.
   Confirm small gains by re-evaluating the current best and candidate with the
   same additional seed, without retraining:

   ```bash
   uv run python evaluation.py --run runs/CURRENT_BEST_ID \
     --name confirm-best --eval-seed 1042 --swanlab-mode online
   uv run python evaluation.py --run runs/CANDIDATE_ID \
     --name confirm-candidate --eval-seed 1042 --swanlab-mode online
   ```

   Ties are discarded. For a one-question gain, require the candidate to also
   beat the current best checkpoint in the paired confirmation. Confirmations
   reuse existing weights, are recorded separately, and do not create another
   candidate attempt. Use at most one confirmation pair per candidate.

8. For an accepted improvement, stage only `experiment.py`, inspect the staged
   diff, and commit locally with the trial, method, and score in the message:

   ```bash
   git add -- experiment.py
   git diff --cached --check
   git diff --cached -- experiment.py
   git commit -m "research: trial 03 prompt improves AIME25 pass@3"
   git rev-parse HEAD
   ```

   First verify there are no unrelated staged changes. Confirm the committed
   file matches the run's `experiment_sha256` and preserve the evaluated method,
   including CLI overrides, in the result record. Update `best.json` to this
   commit and run. For a discarded/failed candidate, restore only the candidate
   file; HEAD remains at the current best commit:

   ```bash
   git restore --source=BEST_COMMIT --staged --worktree -- experiment.py
   ```

   Append every outcome to `decisions.jsonl` using the schema below. Keep all
   result files and confirmations. Never change a result to alter its score or
   run `git reset --hard` / `git clean` on the shared workspace.
9. Repeat until 10 attempts (or a smaller user budget) are recorded or the user stops you.
   Regenerate the progress plot after each decision and at the end:

   ```bash
   uv run python plot_progress.py --output runs/progress.png
   ```

   Report completed, failed, kept, and discarded trials with artifact links.
   Local improvement commits are authorized. Do not push unless requested.

## Decision records

`decisions.jsonl` is append-only, one row per candidate attempt; the baseline is
trial 0 in the figure and is not a decision row. Number candidates consecutively
from 1 through 10. `parent_commit` is the current best commit before the attempt;
the run's `code_commit` must match it. `commit` is the selected best commit after
the decision: the new commit on keep, or `parent_commit` on discard/failed.
Scores always come from the run's result file, not a manually supplied score.

```json
{"baseline_run_id":"BASELINE_RUN_ID","trial":1,"run_id":"CANDIDATE_RUN_ID","decision":"keep","parent_commit":"FULL_PARENT_SHA","commit":"FULL_NEW_SHA","label":"concise reasoning","reason":"Higher pass@3 under the same protocol","confirmation_run_ids":[]}
```

Use `decision="discard"` for a completed candidate that is not accepted and
`decision="failed"` for an unsuccessful attempt. If no run artifact was created,
use `run_id=null` and record the error in `reason`. Failed attempts have no score.

`best.json` starts at the baseline, and changes only on accepted improvements:

```json
{"baseline_run_id":"BASELINE_RUN_ID","run_id":"BEST_RUN_ID","commit":"FULL_BEST_SHA","model_path":null,"pass_at_3":0.0}
```

Replace placeholders and copy the actual score and model path from the result;
`model_path` is null for a prompt-only base-model run. These files, runs, and plots
are Git-ignored so rollback retains the full experiment history. The plot checks
that a series uses the same model, data, fixed utilities, and evaluation protocol
and follows the recorded parent-commit chain.

AIME25 serves as the development benchmark during this search. Save an untouched
benchmark such as AIME26 for a later final evaluation if one is requested.

## SwanLab analysis

The project logs these metrics, which are the default scope for experiment
analysis: `train/reward`, `train/degenerate_fraction`, `train/optimizer_steps`,
`train/completion_tokens`, `train/hit_max_tokens_rate`, `train/step_seconds`,
`trainer/*`, `opd/reverse_kl_mean`, `opd/reverse_kl_std`,
`sft/answer_only_fraction`, `sft/supervised_tokens`,
`eval/pass_at_3`, `eval/average_at_3`, `eval/format_rate`,
`eval/hit_max_tokens_rate`, `eval/completion_tokens`, and `eval/seconds`.

Read the skill's CLI reference. Discover actual column keys, then query explicit
keys and save data to a file. Use default pagination; prefer summaries for quick
comparisons. Substitute the path from the experiment's SwanLab URL:

```bash
uv run swanlab api run columns USER/PROJECT/RUN --page_size 50
uv run swanlab api run summary USER/PROJECT/RUN \
  --keys eval/pass_at_3,eval/average_at_3,eval/format_rate \
  --save runs/RUN_ID/swanlab-summary.json
uv run swanlab api run metrics USER/PROJECT/RUN \
  --keys train/reward,train/degenerate_fraction \
  --ignore-timestamp --save runs/RUN_ID/swanlab-training.json
```

Use the skill's plotting scripts when a chart helps explain a hypothesis.
Distinguish observed measurements from your explanation of why they changed.
Local result files remain the evidence for scores and saved weights.
OPD reverse-KL is a sampled estimate and can be negative. SFT's token counters
measure supervised targets; they do not measure sampled response accuracy.
