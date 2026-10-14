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
        "Trial 01: baseline hit_max_tokens_rate is 0.956 and format_rate 0.044, so almost every "
        "sample is truncated before emitting a boxed answer. Explicitly capping solution length at "
        "350 words and banning headings/bullets should let the model finish and lift pass@3."
    ),
    "prompt": (
        "Solve the problem with a compact, efficient solution.\n"
        "Constraints:\n"
        "- Use at most 350 words. Brevity is essential.\n"
        "- Do not restate the problem.\n"
        "- No headings, no bullet lists, no numbered lists; write plain sentences and short equations.\n"
        "- Show only the essential steps and the arithmetic.\n"
        "- You must finish well before the 1024-token limit.\n"
        "End with a final line of exactly this form: \\boxed{integer}"
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
