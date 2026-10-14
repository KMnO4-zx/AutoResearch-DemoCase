"""The coding agent edits prompts and training settings during research.

Importing this file does not start training or evaluation. The fixed utilities
validate this configuration and own the training and evaluation budgets.
"""

EXPERIMENT = {
    "mode": "prompt",  # Choose "prompt", "grpo", "sft", or "opd".
    # Fixed optimization target. OPD teachers may use other supported models.
    "base_model": "Qwen/Qwen3.5-4B",
    "teacher_base_model": "Qwen/Qwen3.8-27B",  # Used by OPD only.
    "teacher_model_path": None,  # Optional frozen teacher sampler checkpoint.
    "hypothesis": (
        "Trial 02: trial 01 cut truncation 0.956 -> 0.733 but 66/90 samples still never emit an "
        "answer. Among completed samples accuracy is 33% both below and above 400 tokens, so length "
        "buys no accuracy. Tightening the cap from 350 to 200 words should convert most truncated "
        "samples into scored ones without lowering per-sample accuracy."
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
    "learning_rate": 4e-5,
    "lora_rank": 16,
    "train_temperature": 1.0,
    "group_size": 4,  # OPD may use 1; GRPO needs at least 2.
    "opd_kl_coef": 1.0,
    # SFT defaults to DAPO final-answer supervision. Optional local JSONL can
    # contain problem, answer, and a verified assistant solution per row.
    "sft_data": None,
}
