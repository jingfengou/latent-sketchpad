import os


def resolve_qwen35_attn_implementation() -> str:
    override = os.environ.get("LATENT_SKETCHPAD_QWEN35_ATTN_IMPL")
    if override:
        return override
    return "flash_attention_2"
