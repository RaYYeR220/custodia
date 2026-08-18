"""Runtime configuration.

Everything is read from the environment (a local ``.env`` is loaded if present)
so the same code runs from the CLI, the API, the MCP server and the eval harness
without any of them owning settings the others need.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    # --- graph -------------------------------------------------------------
    hydra_uri: str = field(default_factory=lambda: _env("HYDRA_URI", "neo4j://127.0.0.1:7687"))
    hydra_token: str = field(
        default_factory=lambda: _env("HYDRA_TOKEN", "local-development-token-32-bytes")
    )
    corpus: str = field(default_factory=lambda: _env("CUSTODIA_CORPUS", "default"))

    # --- language model ----------------------------------------------------
    #: any OpenAI-compatible endpoint: OpenRouter, Venice, a local server
    llm_base_url: str = field(
        default_factory=lambda: _env("CUSTODIA_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("CUSTODIA_LLM_API_KEY"))
    #: cheap, high-throughput model for turning turns into facts
    extract_model: str = field(
        default_factory=lambda: _env("CUSTODIA_EXTRACT_MODEL", "qwen/qwen3.7-flash")
    )
    #: the model that reads a warrant and writes the answer
    answer_model: str = field(
        default_factory=lambda: _env("CUSTODIA_ANSWER_MODEL", "google/gemini-3-flash-preview")
    )
    #: the grader used by the evaluation harness, kept separate from the
    #: answering model so a run can be judged by something it did not write
    judge_model: str = field(
        default_factory=lambda: _env("CUSTODIA_JUDGE_MODEL", "google/gemini-3-flash-preview")
    )
    #: long-context model for the full-context baseline (needs the whole haystack)
    baseline_model: str = field(
        default_factory=lambda: _env("CUSTODIA_BASELINE_MODEL", "qwen/qwen3.7-flash")
    )
    llm_timeout: float = field(default_factory=lambda: _num("CUSTODIA_LLM_TIMEOUT", 90.0))
    llm_retries: int = field(default_factory=lambda: int(_num("CUSTODIA_LLM_RETRIES", 3)))
    llm_concurrency: int = field(default_factory=lambda: int(_num("CUSTODIA_LLM_CONCURRENCY", 8)))

    # --- cache -------------------------------------------------------------
    #: responses are content-addressed on disk, so a re-run costs nothing and a
    #: shipped cache lets the demo run with no credentials at all
    cache_dir: Path = field(
        default_factory=lambda: Path(_env("CUSTODIA_CACHE_DIR", str(REPO_ROOT / "data" / "cache")))
    )
    cache_only: bool = field(default_factory=lambda: _flag("CUSTODIA_CACHE_ONLY", False))

    # --- retrieval ---------------------------------------------------------
    seed_entities: int = field(default_factory=lambda: int(_num("CUSTODIA_SEED_ENTITIES", 12)))
    seed_lexical: int = field(default_factory=lambda: int(_num("CUSTODIA_SEED_LEXICAL", 24)))
    path_max_len: int = field(default_factory=lambda: int(_num("CUSTODIA_PATH_MAX_LEN", 3)))
    path_count: int = field(default_factory=lambda: int(_num("CUSTODIA_PATH_COUNT", 400)))
    warrant_size: int = field(default_factory=lambda: int(_num("CUSTODIA_WARRANT_SIZE", 20)))

    # --- gate --------------------------------------------------------------
    #: minimum retrieval score for a fact to be allowed into a warrant at all
    evidence_floor: float = field(default_factory=lambda: _num("CUSTODIA_EVIDENCE_FLOOR", 0.05))
    #: run the second-pass entailment check on every citation
    verify_citations: bool = field(default_factory=lambda: _flag("CUSTODIA_VERIFY_CITATIONS", True))

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key) and not self.cache_only

    def cache_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings - used by tests that patch the environment."""
    global _settings
    _settings = None
