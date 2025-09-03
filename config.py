# config.py
"""Configuration and constants for the application"""

import os
from dotenv import load_dotenv

# Load environment variables (override so .env wins in dev)
load_dotenv(override=True)

def _get_key(name: str):
    """Return a valid key or None (reject placeholders/templates)."""
    v = os.getenv(name, "") or ""
    low = v.strip().lower()
    if not v:
        return None
    # treat common templates as missing
    if low.startswith(("your_", "paste_", "sk-xxxx")):
        return None
    return v.strip()

class Config:
    # API keys (sanitized)
    GROQ_API_KEY   = _get_key("GROQ_API_KEY")
    CLAUDE_API_KEY = _get_key("CLAUDE_API_KEY")
    OPENAI_API_KEY = _get_key("OPENAI_API_KEY")
    GEMINI_API_KEY = _get_key("GEMINI_API_KEY")  # if you add Gemini later

    # Model options
    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    # Claude model
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

    # OpenAI models (ensure your generator supports whichever you pick)
    OPENAI_MODELS = [
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    DEFAULT_OPENAI_MODEL = "gpt-4o"

    # Models that use max_completion_tokens instead of max_tokens
    NEWER_OPENAI_MODELS = ["gpt-5", "gpt-4o", "gpt-4o-mini"]

    # Data generation limits
    MAX_ROWS = 1000
    MIN_ROWS = 1
    DEFAULT_ROWS = 50

    # Analysis thresholds
    HIGH_CORRELATION_THRESHOLD = 0.7
    UNIQUE_ID_THRESHOLD = 0.95
    CATEGORICAL_THRESHOLD = 20

    # Analysis options
    ANALYSIS_METHODS = ["code", "llm"]
    COMPARISON_METHODS = ["code", "llm"]
    PROMPT_GENERATION_METHODS = ["code", "llm"]

    # API priority for data generation
    API_PRIORITY = ["claude", "openai", "groq"]
