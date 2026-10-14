"""The coding agent edits prompts and training settings during research.

Importing this file does not start training or evaluation. The fixed utilities
validate this configuration and own the training and evaluation budgets.
"""

EXPERIMENT = {
    "mode": "grpo",  # Choose "prompt", "grpo", "sft", or "opd".
    # Fixed optimization target. OPD teachers may use other supported models.
    "base_model": "Qwen/Qwen3.5-4B",
    "teacher_base_model": "Qwen/Qwen3.8-27B",  # Used by OPD only.
    "teacher_model_path": None,  # Optional frozen teacher sampler checkpoint.
    "hypothesis": "Baseline: concise mathematical reasoning with outcome-only GRPO.",
    "prompt": (
        "Solve the problem carefully using concise reasoning. "
        "Put your final integer answer inside \\boxed{}."
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
