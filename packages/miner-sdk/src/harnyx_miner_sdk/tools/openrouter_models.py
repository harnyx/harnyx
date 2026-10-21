"""Shared miner eligibility rules for native OpenRouter model IDs."""


def is_free_openrouter_model(model: str) -> bool:
    parts = model.strip().lower().split(":")
    return parts[0] == "openrouter/free" or "free" in parts[1:]


def validate_paid_openrouter_model(model: str) -> str:
    """Validate native routing syntax, preserving paid promotions at zero price."""
    value = model.strip()
    if is_free_openrouter_model(value):
        raise ValueError("explicitly free OpenRouter models are not allowed")
    if "/" not in value or any(char.isspace() for char in value) or any(c in value for c in "?#\\"):
        raise ValueError("OpenRouter model must be a native model id")
    return value
