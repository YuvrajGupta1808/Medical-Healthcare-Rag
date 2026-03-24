from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Root of the repo — two levels up from this file (src/utils/ → src/ → root)
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str = "v1") -> str:
    """
    Load and cache a versioned prompt file.

    Args:
        name: prompt filename without extension (e.g. "system", "citation")
        version: prompt version directory (e.g. "v1")

    Returns:
        Stripped prompt string.

    Raises:
        FileNotFoundError: if the prompt file does not exist.
    """
    path = _PROMPTS_DIR / version / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Expected at prompts/{version}/{name}.txt"
        )
    return path.read_text(encoding="utf-8").strip()
