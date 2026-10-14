# Project guidance for coding agents

Follow [program.md](program.md) to drive the AutoResearch workflow.

- Read [README.md](README.md) and [program.md](program.md) before working.
- Any coding agent that can edit files and execute commands can drive this case.
- Each coding agent uses its assigned branch and a separate worktree/clone.
  Keep run artifacts, decisions, and best.json local to that worktree. Prepared
  datasets may be shared read-only; evaluation settings must match for comparisons.
- `pytrio-skill` and `swanlab-skill` are explicitly allowed for this project.
  Use their instructions to write correct SDK code and analyze experiment data.
- The objective is to improve `Qwen/Qwen3.5-4B` AIME25 pass@3 over its initial
  prompt baseline. Keep the student fixed at 4B. Prompt optimization, GRPO,
  SFT, and OPD are supported; OPD may use any compatible PyTRIO-supported teacher.
- Run one baseline and at most 10 candidate attempts. Prompt-only changes are
  first-class candidates and perform zero optimizer updates. Failed and
  discarded attempts count toward the same 10-attempt budget.
- During autonomous search, edit only `experiment.py`. Keep `prepare.py`, `evaluation.py`, `plot_progress.py`,
  datasets, grading, and the evaluation protocol fixed. This restriction applies
  to research trials; user-requested maintenance may edit the
  project implementation.
- Each training candidate starts from a fresh base-model LoRA and uses at most
  20 optimizer updates. Never continue training from the previous candidate.
  Use `mode="prompt"` for prompt-only evaluation on the base model.
- Evaluate all 30 AIME25 questions with three samples per question, 1024 output
  tokens per sample, and at most 16 problem requests in flight. Distinguish pass@3 from average@3.
- Set evaluation thinking with `--eval-thinking` or `--no-eval-thinking` before
  the baseline, then keep it fixed for all comparable candidates. New runs
  default to off; re-evaluation inherits the source setting. Thinking tokens
  count toward the same 1024-token budget.
- Local tests may mock the model service in
  `tests/`; do not add fake outputs or smoke-only branches to production code.
- Keep credentials in the SDK login store or environment. Do not copy them into
  experiment files, logs, committed files, or command arguments.
- Start from a committed baseline. Commit `experiment.py` after each accepted
  pass@3 improvement, and base the next change on that commit. On no improvement
  or failure, restore only `experiment.py` from the latest accepted commit.
  Append every outcome to `decisions.jsonl` and update `best.json` on acceptance.
  Keep run artifacts for plotting. Never reset unrelated work or push without
  a separate user instruction.
