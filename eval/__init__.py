"""Custodia's evaluation harness: datasets, baselines, scorers, poison suite.

The harness is deliberately decoupled from ``src/custodia``. Custodia is imported
lazily, inside the functions that need it, for two reasons: the harness must stay
usable while the memory layer is still being assembled, and a judge should be
able to score the *baselines* even when the graph is unreachable.

One rule governs everything in this package: a number that was not measured is
never printed as if it were. Failures surface as errors or as ``None`` (rendered
``not measured``); they never quietly become a passing score.

This module holds the two things every submodule shares -- the token estimator
and the language-model binding -- so neither ``baselines`` nor ``scorers`` has to
import the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "TOKEN_ESTIMATOR",
    "estimate_tokens",
    "ChatLLM",
    "LlmBinding",
    "NoProviderConfigured",
    "resolve_llm",
]

# --------------------------------------------------------------------------- #
# token accounting
# --------------------------------------------------------------------------- #

#: Every ``prompt_tokens`` figure in this harness comes from this estimator, not
#: from a provider usage field. A real tokenizer would pull in a model-specific
#: dependency and would still be wrong for whichever model the run actually used,
#: so we use one crude estimator consistently and say so in every report. The
#: figure is comparable *between systems in the same run*, which is the only
#: comparison it is used for (warrant size vs. full haystack).
TOKEN_ESTIMATOR = "chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact)"


def estimate_tokens(text: str) -> int:
    """Approximate token count for prompt-size comparisons.

    Deliberately dependency-free: see :data:`TOKEN_ESTIMATOR` for why an
    approximate, uniformly-applied estimate beats a precise but model-specific
    one here.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# --------------------------------------------------------------------------- #
# language model binding
# --------------------------------------------------------------------------- #


class NoProviderConfigured(RuntimeError):
    """No usable language model. Raised instead of degrading to a fake answer.

    Every runner catches this and stops with a readable message. Nothing in the
    harness substitutes a canned response for a model call, because a canned
    response would silently become a benchmark number.
    """


@runtime_checkable
class ChatLLM(Protocol):
    """The single-turn completion surface the harness needs from any provider."""

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        ...


@dataclass(slots=True)
class LlmBinding:
    """A resolved model, plus where it came from.

    ``origin`` is recorded in every result file. Custodia and the baselines must
    share a model for the comparison to mean anything, so a report that cannot
    show which binding answered is a report a judge cannot trust.
    """

    client: ChatLLM
    model: str
    origin: str  # "custodia.llm" | "eval-builtin"
    calls: int = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.calls += 1
        return self.client.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )

    def provenance(self) -> dict[str, Any]:
        return {"model": self.model, "binding": self.origin, "calls": self.calls}


def _settings() -> Any:
    """Custodia's settings if importable, else a minimal env-backed stand-in."""
    try:
        from custodia.config import settings

        return settings()
    except Exception:  # pragma: no cover - only when custodia.config is absent
        import os

        @dataclass(slots=True)
        class _Fallback:
            llm_base_url: str = os.environ.get(
                "CUSTODIA_LLM_BASE_URL", "https://openrouter.ai/api/v1"
            )
            llm_api_key: str = os.environ.get("CUSTODIA_LLM_API_KEY", "")
            answer_model: str = os.environ.get(
                "CUSTODIA_ANSWER_MODEL", "google/gemini-2.5-flash"
            )
            extract_model: str = os.environ.get(
                "CUSTODIA_EXTRACT_MODEL", "google/gemini-2.5-flash"
            )
            llm_timeout: float = 90.0

        return _Fallback()


def _call_flexible(fn: Callable[..., Any], prompt: str, **kwargs: Any) -> str:
    """Call a provider entry point whose exact signature we do not control.

    ``custodia.llm`` is being written concurrently with this harness. Rather than
    hard-code one spelling and break on the first mismatch, we drop optional
    keywords one at a time until the call is accepted, and let a genuine provider
    failure propagate untouched.
    """
    optional = ["max_tokens", "temperature", "system", "model"]
    attempt = dict(kwargs)
    while True:
        try:
            result = fn(prompt, **attempt)
            break
        except TypeError as exc:
            dropped = next((k for k in optional if k in attempt), None)
            if dropped is None or "argument" not in str(exc):
                raise
            attempt.pop(dropped)
    if isinstance(result, str):
        return result
    for attr in ("text", "content", "message"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(result, dict):
        for key in ("text", "content", "answer"):
            if isinstance(result.get(key), str):
                return str(result[key])
    return str(result)


@dataclass(slots=True)
class _BoundCallable:
    """Adapts a bare callable found on ``custodia.llm`` to :class:`ChatLLM`."""

    fn: Callable[..., Any]
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return _call_flexible(
            self.fn,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model,
        )


@dataclass(slots=True)
class _OpenAICompatible:
    """Minimal OpenAI-compatible chat client, used only when Custodia has none.

    This is a transport, not a second provider: it reads the same base URL, key
    and model out of Custodia's settings, so "the baselines share Custodia's
    model" stays true. Reports label the binding ``eval-builtin`` so the
    difference is never invisible.
    """

    base_url: str
    api_key: str
    model: str
    timeout: float = 90.0
    retries: int = 3

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        import httpx

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": 0,  # honoured by some providers; harmless where it is not
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last: Exception | None = None
        for _ in range(max(1, self.retries)):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                body = response.json()
                return str(body["choices"][0]["message"]["content"] or "")
            except Exception as exc:  # retried; the last one is re-raised
                last = exc
        raise RuntimeError(f"language model call failed: {last}") from last


def resolve_llm(model: str | None = None, *, role: str = "answer") -> LlmBinding:
    """Bind to the language model the whole run will share.

    Preference order is deliberate. Custodia's own client wins because a shared
    client means a shared cache and identical decoding settings across Custodia
    and the baselines. The built-in transport is the fallback so that the harness
    remains runnable on its own; both are labelled in the output.

    Raises :class:`NoProviderConfigured` -- never returns a stub -- when no key is
    configured.

    The contract expected of ``custodia.llm`` is one of:

    * a module-level ``complete(prompt, *, system, temperature, max_tokens, model)``
      or ``chat(...)`` returning the reply text;
    * a class ``LlmClient`` / ``LLM`` whose instances expose ``complete`` or ``chat``
      with the same shape.
    """
    config = _settings()
    default = config.answer_model if role == "answer" else config.extract_model
    chosen = model or default

    bound = _bind_custodia(chosen)
    if bound is not None:
        return bound

    key = getattr(config, "llm_api_key", "")
    if not key:
        raise NoProviderConfigured(
            "no provider configured: set CUSTODIA_LLM_API_KEY (and optionally "
            "CUSTODIA_LLM_BASE_URL / CUSTODIA_ANSWER_MODEL), or expose a "
            "complete()/chat() entry point from custodia.llm"
        )
    client = _OpenAICompatible(
        base_url=getattr(config, "llm_base_url", "https://openrouter.ai/api/v1"),
        api_key=key,
        model=chosen,
        timeout=float(getattr(config, "llm_timeout", 90.0)),
    )
    return LlmBinding(client=client, model=chosen, origin="eval-builtin")


def _bind_custodia(model: str) -> LlmBinding | None:
    """Find a usable entry point on ``custodia.llm``; ``None`` if there is none."""
    try:
        from custodia import llm as module  # noqa: PLC0415 - lazy on purpose
    except Exception:
        return None

    for name in ("LlmClient", "LLM", "Client", "ChatClient"):
        factory = getattr(module, name, None)
        if not callable(factory):
            continue
        try:
            instance = factory(model=model)
        except TypeError:
            try:
                instance = factory()
            except Exception:
                continue
        except Exception:
            continue
        method = getattr(instance, "complete", None) or getattr(instance, "chat", None)
        if callable(method):
            return LlmBinding(
                client=_BoundCallable(method, model), model=model, origin="custodia.llm"
            )

    for name in ("complete", "chat", "chat_completion"):
        fn = getattr(module, name, None)
        if callable(fn):
            return LlmBinding(
                client=_BoundCallable(fn, model), model=model, origin="custodia.llm"
            )
    return None
