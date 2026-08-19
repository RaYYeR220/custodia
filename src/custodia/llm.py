"""Provider-agnostic access to an OpenAI-compatible chat endpoint.

Custodia runs against OpenRouter and Venice, which differ only in base URL and
model names, so this client targets ``POST /chat/completions`` and nothing else.
Three decisions here are load-bearing rather than incidental:

* **Every response is content-addressed on disk.** The key covers the model, the
  messages and the sampling parameters, so a hit is exactly the call that would
  otherwise have gone out, and it never touches the network. Re-running an
  ingest is free, and shipping a populated cache is what lets the demo run end
  to end with no credentials at all. ``cache_only`` turns a miss into a failure
  instead of a surprise bill.
* **The transport is injectable.** Retry, caching, pacing and JSON repair are the
  parts of this module worth testing and none of them should need a network to
  test. The default transport is httpx; tests pass a callable.
* **Outbound calls are paced client-side.** The provider meters us, and a 429 the
  retries cannot outlast arrives at the gate as "no evidence" - so an unpaced run
  measures the rate limiter instead of the product. A token bucket sits in front
  of the transport, and ``usage()`` reports what it cost, because a throttled run
  has to be reported as one.
* **Every failure collapses into one exception.** A provider error, a timeout,
  unparseable JSON and a miss under ``cache_only`` all raise
  :class:`LLMUnavailable`, because everything upstream treats them identically:
  the failure path and the "I don't know" path are the same path, and that is
  only true if a caller cannot accidentally distinguish them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from custodia import config
from custodia.prompts import JSON_REPAIR

log = logging.getLogger("custodia.llm")

#: first backoff pause, doubled per attempt
RETRY_BASE_DELAY = 0.75

#: how much of an unparseable reply is echoed back in the repair turn
REPAIR_ECHO = 4000

#: a provider is allowed to ask us to wait, but not to hang the run
MAX_RETRY_AFTER = 60.0


class LLMUnavailable(RuntimeError):
    """No usable completion: provider error, timeout, bad JSON, or a cache miss
    while running cache-only."""


class LLMTransportError(RuntimeError):
    """A request that never reached the provider - connection reset, DNS, TLS."""


@dataclass(slots=True)
class LLMRequest:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]
    timeout: float


@dataclass(slots=True)
class LLMResponse:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


Transport = Callable[[LLMRequest], LLMResponse]


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---- cache ---------------------------------------------------------------- #


def cache_key(
    model: str,
    messages: Sequence[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> str:
    """Content address of a call, stable across processes and machines."""
    blob = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "temperature": round(float(temperature), 4),
            "max_tokens": int(max_tokens),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- transport ------------------------------------------------------------ #


def _httpx_transport(timeout: float) -> Transport:
    client = httpx.Client(timeout=timeout, follow_redirects=True)

    def send(request: LLMRequest) -> LLMResponse:
        try:
            response = client.post(
                request.url,
                headers=request.headers,
                json=request.payload,
                timeout=request.timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMTransportError(f"timeout after {request.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMTransportError(str(exc)) from exc
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return LLMResponse(response.status_code, body, dict(response.headers))

    send.close = client.close  # type: ignore[attr-defined]
    return send


def _wait(seconds: float) -> None:
    """The one place this module blocks, so a test can drive it without waiting."""
    if seconds > 0:
        time.sleep(seconds)


def _retry_after(headers: Mapping[str, str], fallback: float) -> float:
    """Seconds the provider asked us to wait, in either form the RFC allows."""
    raw = ""
    for name, value in headers.items():
        if name.lower() == "retry-after":
            raw = str(value).strip()
            break
    if not raw:
        return fallback
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return fallback
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, min(seconds, MAX_RETRY_AFTER))


class TokenBucket:
    """Requests per minute, shared by every thread on one client.

    A bucket rather than a fixed delay because the traffic is bursty by nature:
    a batch of extraction windows arrives all at once, and the first few should
    go straight out. A caller that finds the bucket empty reserves its permit
    before sleeping, so concurrent callers queue in order instead of waking up
    together and colliding again.
    """

    def __init__(
        self,
        per_minute: float,
        *,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = _wait,
    ) -> None:
        self.rate = max(0.0, per_minute) / 60.0
        self.capacity = burst if burst is not None else max(1.0, min(per_minute / 10.0, 10.0))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Take one permit, blocking if there is none. Returns seconds waited."""
        if self.rate <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            deficit = 1.0 - self._tokens
            # the permit is taken either way; a negative balance is the queue
            self._tokens -= 1.0
        if deficit <= 0:
            return 0.0
        pause = deficit / self.rate
        self._sleep(pause)
        return pause


def _excerpt(body: Any, limit: int = 300) -> str:
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)[:limit]
    return text[:limit].replace("\n", " ")


# ---- JSON salvage --------------------------------------------------------- #

_FENCE = re.compile(r"```[A-Za-z0-9_-]*\s*\n?(.*?)```", re.S)
_OPENERS = {"{": "}", "[": "]"}


def _unfence(text: str) -> str:
    match = _FENCE.search(text)
    return match.group(1) if match else text


def _outermost(text: str) -> str | None:
    """The first balanced ``{...}`` or ``[...]``, ignoring braces inside strings."""
    start = next((i for i, ch in enumerate(text) if ch in _OPENERS), -1)
    if start < 0:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in _OPENERS:
            stack.append(_OPENERS[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[start : i + 1]
    return None


def loads(text: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort JSON out of a chat reply, or ``None``.

    Only whole values are accepted. A half-parsed object is worse than nothing
    here: it would look like evidence while being a guess.
    """
    stripped = _unfence(text).strip()
    for blob in (stripped, _outermost(stripped)):
        if not blob:
            continue
        try:
            value = json.loads(blob)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


# ---- client --------------------------------------------------------------- #


class LLM:
    """Chat and JSON calls against one OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: config.Settings | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.settings = settings or config.settings()
        self._transport = transport
        # an injected transport stands in for credentials: tests and offline
        # harnesses drive the live path without a key on the machine
        self._injected = transport is not None
        self._lock = threading.Lock()
        #: one limiter per client, shared by every thread `batch_json` starts
        self.limiter = TokenBucket(self.settings.llm_rpm)
        self._usage = {
            "calls": 0,
            "cached": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "throttled": 0,
            "waited_ms": 0,
        }

    # ------------------------------------------------------------------ state

    @property
    def enabled(self) -> bool:
        """Whether a live call is possible at all. Cache hits work regardless."""
        if self.settings.cache_only:
            return False
        return self._injected or bool(self.settings.llm_api_key)

    def usage(self) -> dict[str, int]:
        """Running totals for the run.

        ``waited_ms`` counts every millisecond spent *not* sending - client-side
        pacing and a provider's own ``Retry-After`` alike - and ``throttled``
        counts the 429s behind it. A report that omits them can describe a run
        that spent half its time queueing as if it were a measurement of the
        system. It is summed across worker threads, so under ``batch_json`` it
        is thread-time and will exceed the wall clock rather than track it.
        """
        with self._lock:
            return dict(self._usage)

    def close(self) -> None:
        closer = getattr(self._transport, "close", None)
        if closer is not None:
            closer()

    # ------------------------------------------------------------------ calls

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        model = model or self.settings.extract_model
        # `stop` stays out of the key: Custodia never varies it for the same
        # prompt, and a key that covers only the four documented fields is what
        # makes a cache directory portable between checkouts.
        key = cache_key(model, messages, temperature, max_tokens)

        hit = self._read_cache(key)
        if hit is not None:
            self._record(hit, cached=True)
            return hit
        if self.settings.cache_only:
            raise LLMUnavailable(f"cache miss under cache_only (key {key[:12]})")
        if not self.enabled:
            raise LLMUnavailable("no LLM credentials configured")

        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if stop:
            payload["stop"] = list(stop)
        payload.update(_provider_extras(self.settings.llm_base_url))

        completion = _completion(self._post(payload), model)
        self._write_cache(key, completion)
        self._record(completion, cached=False)
        return completion

    def json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """A completion parsed as JSON, with one repair attempt.

        A bare array comes back wrapped under ``items`` so callers always get a
        mapping and never have to type-check the top level.
        """
        completion = self.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        parsed = loads(completion.text)
        if parsed is None:
            repair = [
                *messages,
                {"role": "assistant", "content": completion.text[:REPAIR_ECHO]},
                {"role": "user", "content": JSON_REPAIR},
            ]
            completion = self.chat(
                repair, model=model, temperature=temperature, max_tokens=max_tokens
            )
            parsed = loads(completion.text)
        if parsed is None:
            raise LLMUnavailable("model did not return parseable JSON")
        return parsed if isinstance(parsed, dict) else {"items": parsed}

    def batch_json(
        self,
        batches: list[list[dict[str, Any]]],
        *,
        model: str | None = None,
        **kw: Any,
    ) -> list[dict[str, Any] | None]:
        """Run many JSON calls concurrently, preserving input order.

        One failed item yields ``None``. A window that the model choked on must
        not cost us the rest of the corpus - the caller falls back per item.
        """
        if not batches:
            return []
        results: list[dict[str, Any] | None] = [None] * len(batches)
        workers = max(1, min(int(self.settings.llm_concurrency), len(batches)))
        if workers == 1:
            for i, batch in enumerate(batches):
                results[i] = self._try_json(batch, model=model, **kw)
            return results
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="custodia-llm") as pool:
            futures = {
                pool.submit(self._try_json, batch, model=model, **kw): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def _try_json(self, batch: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
        try:
            return self.json(batch, **kw)
        except LLMUnavailable as exc:
            log.debug("batch item unavailable: %s", exc)
        except Exception:  # noqa: BLE001 - one bad item must not lose the batch
            log.warning("batch item failed", exc_info=True)
        return None

    # ------------------------------------------------------------------- http

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "X-Title": "Custodia",
        }
        request = LLMRequest(url, headers, payload, self.settings.llm_timeout)

        attempts = max(1, int(self.settings.llm_retries))
        reason = "no attempt made"
        for attempt in range(attempts):
            pause = RETRY_BASE_DELAY * (2**attempt)
            # every attempt is a request against the quota, retries included
            self._paced()
            try:
                response = self._send(request)
            except (LLMTransportError, TimeoutError, OSError) as exc:
                reason = f"transport: {exc}"
            else:
                if response.status < 300:
                    if isinstance(response.body, dict):
                        return response.body
                    raise LLMUnavailable(f"non-JSON response body: {_excerpt(response.body)}")
                reason = f"http {response.status}: {_excerpt(response.body)}"
                if response.status == 429:
                    # being metered is ordinary operation, not a fault: the
                    # provider is telling us the pace, so take the pace it gives
                    self._count("throttled", 1)
                    pause = _retry_after(response.headers, pause)
                    log.debug("rate limited, pausing %.2fs before retry %d", pause, attempt + 1)
                elif response.status < 500:
                    raise LLMUnavailable(reason)
            if attempt + 1 < attempts:
                self._count("waited_ms", int(pause * 1000))
                _wait(pause)
        raise LLMUnavailable(f"provider unavailable after {attempts} attempts - {reason}")

    def _paced(self) -> None:
        waited = self.limiter.acquire()
        if waited:
            self._count("waited_ms", int(waited * 1000))

    def _send(self, request: LLMRequest) -> LLMResponse:
        if self._transport is None:
            self._transport = _httpx_transport(self.settings.llm_timeout)
        return self._transport(request)

    # ------------------------------------------------------------------ cache

    def _cache_file(self, key: str, root: Path | None = None) -> Path:
        return Path(root or self.settings.cache_dir) / "llm" / key[:2] / f"{key}.json"

    def _cache_roots(self) -> list[Path]:
        """Writable cache first, then any read-only roots shipped with the repo.

        The shipped root is what lets someone clone this and run the whole
        walkthrough without credentials: the exact completions the demo needs
        are committed, content-addressed by the call that produced them.
        """
        roots = [Path(self.settings.cache_dir)]
        for extra in self.settings.cache_seed_dirs:
            path = Path(extra)
            if path not in roots:
                roots.append(path)
        return roots

    def _read_cache(self, key: str) -> Completion | None:
        record = None
        for root in self._cache_roots():
            try:
                record = json.loads(self._cache_file(key, root).read_text("utf-8"))
                break
            except (OSError, ValueError):
                continue
        if record is None:
            return None
        text = record.get("text")
        if not isinstance(text, str):
            return None
        return Completion(
            text=text,
            model=str(record.get("model", "")),
            cached=True,
            prompt_tokens=int(record.get("prompt_tokens", 0)),
            completion_tokens=int(record.get("completion_tokens", 0)),
        )

    def _write_cache(self, key: str, completion: Completion) -> None:
        path = self._cache_file(key)
        record = {
            "text": completion.text,
            "model": completion.model,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "written_at": int(time.time()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # a concurrent batch can race on the same key; replace is atomic
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), "utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not cache completion %s: %s", key[:12], exc)

    def _count(self, name: str, amount: int) -> None:
        with self._lock:
            self._usage[name] += amount

    def _record(self, completion: Completion, *, cached: bool) -> None:
        with self._lock:
            self._usage["calls"] += 1
            if cached:
                # a hit costs nothing, so counting its stored tokens again would
                # overstate what the run actually spent
                self._usage["cached"] += 1
            else:
                self._usage["prompt_tokens"] += completion.prompt_tokens
                self._usage["completion_tokens"] += completion.completion_tokens


def _provider_extras(base_url: str) -> dict[str, Any]:
    """Provider-specific body fields needed to keep a prompt actually clean.

    Venice prepends its own system prompt to every request unless asked not to.
    Custodia's whole claim is that the answering model sees the warrant and
    nothing else, so an injected preamble is not a cosmetic difference - it is a
    confound in the one measurement that matters. Turn it off.

    Thinking is disabled for the same reason plus a practical one: a reasoning
    model can spend the whole token budget on a monologue and return an empty
    message, which is indistinguishable from a provider failure downstream.
    """
    if "venice.ai" in base_url:
        return {
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "disable_thinking": True,
                "strip_thinking_response": True,
            }
        }
    return {}


def _completion(body: dict[str, Any], model: str) -> Completion:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMUnavailable(f"provider returned no choices: {_excerpt(body)}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    text = message.get("content", first.get("text"))
    if not isinstance(text, str):
        raise LLMUnavailable(f"provider returned no text: {_excerpt(body)}")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return Completion(
        text=text,
        model=str(body.get("model") or model),
        cached=False,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
