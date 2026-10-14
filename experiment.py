"""The coding agent edits prompts and training settings during research.

Importing this file does not start training or evaluation. The fixed utilities
validate this configuration and own the training and evaluation budgets.
"""

EXPERIMENT = {
    "mode": "sft",  # Choose "prompt", "grpo", "sft", or "opd".
    # Fixed optimization target. OPD teachers may use other supported models.
    "base_model": "Qwen/Qwen3.5-4B",
    "teacher_base_model": "Qwen/Qwen3.8-27B",  # Used by OPD only.
    "teacher_model_path": None,  # Optional frozen teacher sampler checkpoint.
    "hypothesis": (
        "Trial 06: GRPO (trial 04) and OPD (trial 03) both failed because 20 optimizer updates give "
        "too little signal from sparse outcome rewards or teacher matching. Switch to rejection-sampled "
        "SFT: 512 DAPO problems were sampled with the trial-02 brief prompt and only correct, "
        "self-terminating solutions under 600 tokens were kept (150 rows, mean 261 tokens). Every "
        "target therefore teaches exactly the missing behaviour - reason briefly, emit \\boxed{}, stop. "
        "Supervision here is dense (every target token) instead of one bit per trajectory."
    ),
    "prompt": (
        "Solve the problem as briefly as possible. An unfinished solution scores zero.\n"
        "Hard rules:\n"
        "- At most 200 words total; most problems need only 100-150.\n"
        "- Do not restate the problem, and do not use headings or bullet lists.\n"
        "- Give only the key steps and the arithmetic.\n"
        "- Stop immediately after the final line.\n"
        "Final line, with nothing after it: \\boxed{integer}"
    ),
    "steps": 20,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "train_temperature": 1.0,
    "group_size": 4,  # OPD may use 1; GRPO needs at least 2.
    "opd_kl_coef": 1.0,
    # SFT defaults to DAPO final-answer supervision. Optional local JSONL can
    # contain problem, answer, and a verified assistant solution per row.
    "sft_data": "data/rft_solutions.jsonl",  # Rejection-sampled, verified, concise solutions.
}
