"""The LLM client, driven through an injected transport - no network, no key."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import pytest

from custodia import llm as llm_module
from custodia.config import Settings
from custodia.llm import (
    LLM,
    LLMResponse,
    LLMTransportError,
    LLMUnavailable,
    TokenBucket,
)
from custodia.prompts import JSON_REPAIR


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "RETRY_BASE_DELAY", 0.0)


def make_settings(tmp_path, **overrides: Any) -> Settings:
    cfg = Settings()
    cfg.cache_dir = tmp_path / "cache"
    cfg.cache_only = False
    cfg.llm_api_key = "test-key"
    cfg.llm_base_url = "https://example.invalid/v1"
    cfg.llm_timeout = 5.0
    cfg.llm_retries = 3
    cfg.llm_concurrency = 4
    # pinned so a local .env can never make the suite sleep for real
    cfg.llm_rpm = 6000.0
    cfg.extract_model = "test/model"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def body(text: str, prompt: int = 11, completion: int = 7) -> dict[str, Any]:
    return {
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class Recorder:
    """Replays a queued script of responses; the last one repeats."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> LLMResponse:
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def calls(self) -> int:
        return len(self.requests)


HELLO = [{"role": "user", "content": "hello"}]


# ---- cache ---------------------------------------------------------------- #


def test_cache_hit_never_reaches_the_transport(tmp_path):
    transport = Recorder(LLMResponse(200, body("first")))
    client = LLM(make_settings(tmp_path), transport=transport)

    live = client.chat(HELLO)
    cached = client.chat(HELLO)

    assert transport.calls == 1
    assert live.cached is False and cached.cached is True
    assert cached.text == live.text == "first"


def test_cache_is_shared_between_clients(tmp_path):
    settings = make_settings(tmp_path)
    LLM(settings, transport=Recorder(LLMResponse(200, body("stored")))).chat(HELLO)

    cold = Recorder(LLMResponse(500, {"error": "should not be called"}))
    again = LLM(settings, transport=cold).chat(HELLO)

    assert cold.calls == 0
    assert again.cached is True and again.text == "stored"


def test_cache_key_covers_the_sampling_parameters(tmp_path):
    transport = Recorder(LLMResponse(200, body("a")), LLMResponse(200, body("b")))
    client = LLM(make_settings(tmp_path), transport=transport)

    client.chat(HELLO, temperature=0.0)
    client.chat(HELLO, temperature=0.7)

    assert transport.calls == 2


def test_cache_only_turns_a_miss_into_a_failure(tmp_path):
    transport = Recorder(LLMResponse(200, body("never")))
    client = LLM(make_settings(tmp_path, cache_only=True), transport=transport)

    with pytest.raises(LLMUnavailable):
        client.chat(HELLO)
    assert transport.calls == 0
    assert client.enabled is False


def test_cache_only_still_serves_a_hit(tmp_path):
    warm = make_settings(tmp_path)
    LLM(warm, transport=Recorder(LLMResponse(200, body("shipped")))).chat(HELLO)

    offline = LLM(make_settings(tmp_path, cache_only=True, llm_api_key=""))
    result = offline.chat(HELLO)

    assert result.cached is True and result.text == "shipped"


def test_without_credentials_a_live_call_is_refused(tmp_path):
    client = LLM(make_settings(tmp_path, llm_api_key=""))

    assert client.enabled is False
    with pytest.raises(LLMUnavailable):
        client.chat(HELLO)


# ---- retries -------------------------------------------------------------- #


def test_rate_limit_and_server_error_are_retried(tmp_path):
    transport = Recorder(
        LLMResponse(429, {"error": "slow down"}),
        LLMResponse(503, {"error": "restarting"}),
        LLMResponse(200, body("third time")),
    )
    client = LLM(make_settings(tmp_path), transport=transport)

    assert client.chat(HELLO).text == "third time"
    assert transport.calls == 3


def test_timeout_is_retried(tmp_path):
    transport = Recorder(LLMTransportError("timeout after 5.0s"), LLMResponse(200, body("ok")))
    client = LLM(make_settings(tmp_path), transport=transport)

    assert client.chat(HELLO).text == "ok"
    assert transport.calls == 2


def test_client_error_is_fatal_immediately(tmp_path):
    transport = Recorder(LLMResponse(401, {"error": "bad key"}))
    client = LLM(make_settings(tmp_path), transport=transport)

    with pytest.raises(LLMUnavailable, match="401"):
        client.chat(HELLO)
    assert transport.calls == 1


def test_retries_are_bounded(tmp_path):
    transport = Recorder(LLMResponse(500, {"error": "down"}))
    client = LLM(make_settings(tmp_path, llm_retries=2), transport=transport)

    with pytest.raises(LLMUnavailable):
        client.chat(HELLO)
    assert transport.calls == 2


def test_a_reply_without_text_is_unavailable(tmp_path):
    transport = Recorder(LLMResponse(200, {"model": "test/model", "choices": []}))
    client = LLM(make_settings(tmp_path), transport=transport)

    with pytest.raises(LLMUnavailable):
        client.chat(HELLO)


# ---- JSON ----------------------------------------------------------------- #


def test_json_survives_fences_and_prose(tmp_path):
    reply = 'Sure, here you go:\n```json\n{"facts": [{"text": "one"}]}\n```\nHope that helps.'
    client = LLM(make_settings(tmp_path), transport=Recorder(LLMResponse(200, body(reply))))

    assert client.json(HELLO) == {"facts": [{"text": "one"}]}


def test_json_finds_the_outermost_object(tmp_path):
    reply = 'Thinking... {"a": {"b": "}"}, "c": [1, 2]} done'
    client = LLM(make_settings(tmp_path), transport=Recorder(LLMResponse(200, body(reply))))

    assert client.json(HELLO) == {"a": {"b": "}"}, "c": [1, 2]}


def test_json_wraps_a_bare_array(tmp_path):
    client = LLM(make_settings(tmp_path), transport=Recorder(LLMResponse(200, body('[{"n": 1}]'))))

    assert client.json(HELLO) == {"items": [{"n": 1}]}


def test_json_repairs_once(tmp_path):
    transport = Recorder(
        LLMResponse(200, body("I cannot express that as JSON, but the answer is 4.")),
        LLMResponse(200, body('{"answer": 4}')),
    )
    client = LLM(make_settings(tmp_path), transport=transport)

    assert client.json(HELLO) == {"answer": 4}
    assert transport.calls == 2
    repair = transport.requests[1].payload["messages"]
    assert repair[-1]["content"] == JSON_REPAIR
    assert repair[-2]["role"] == "assistant"


def test_json_gives_up_rather_than_guessing(tmp_path):
    transport = Recorder(LLMResponse(200, body("still prose {oops")))
    client = LLM(make_settings(tmp_path), transport=transport)

    with pytest.raises(LLMUnavailable):
        client.json(HELLO)
    assert transport.calls == 2


# ---- batching and accounting ---------------------------------------------- #


def test_batch_isolates_a_failed_item(tmp_path):
    def transport(request: Any) -> LLMResponse:
        prompt = request.payload["messages"][-1]["content"]
        if prompt == "two":
            return LLMResponse(500, {"error": "boom"})
        return LLMResponse(200, body(json.dumps({"said": prompt})))

    client = LLM(make_settings(tmp_path, llm_retries=1), transport=transport)
    out = client.batch_json([[{"role": "user", "content": w}] for w in ("one", "two", "three")])

    assert out == [{"said": "one"}, None, {"said": "three"}]


def test_batch_of_nothing_is_nothing(tmp_path):
    client = LLM(make_settings(tmp_path), transport=Recorder(LLMResponse(200, body("x"))))
    assert client.batch_json([]) == []


# ---- pacing --------------------------------------------------------------- #


class Clock:
    """A hand-driven clock: sleeping advances it, nothing else does."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def tick(self, seconds: float) -> None:
        self.now += seconds


def test_the_bucket_lets_a_burst_through_then_paces():
    clock = Clock()
    bucket = TokenBucket(60.0, burst=3, clock=lambda: clock.now, sleep=clock.sleep)

    assert [bucket.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.acquire() == pytest.approx(1.0)
    assert bucket.acquire() == pytest.approx(1.0)
    assert clock.slept == [1.0, 1.0]


def test_the_bucket_refills_over_time():
    clock = Clock()
    bucket = TokenBucket(60.0, burst=1, clock=lambda: clock.now, sleep=clock.sleep)

    assert bucket.acquire() == 0.0
    clock.tick(5.0)
    assert bucket.acquire() == 0.0
    assert clock.slept == []


def test_pacing_can_be_switched_off():
    bucket = TokenBucket(0.0, sleep=lambda s: pytest.fail("must not sleep"))
    assert [bucket.acquire() for _ in range(50)] == [0.0] * 50


def test_the_limiter_paces_live_calls_but_not_cache_hits(tmp_path):
    transport = Recorder(LLMResponse(200, body("paced")))
    client = LLM(make_settings(tmp_path), transport=transport)
    clock = Clock()
    client.limiter = TokenBucket(60.0, burst=1, clock=lambda: clock.now, sleep=clock.sleep)

    client.chat(HELLO)
    client.chat([{"role": "user", "content": "second"}])
    client.chat(HELLO)  # cached: never leaves the process, never takes a permit

    assert clock.slept == [pytest.approx(1.0)]
    assert client.usage()["waited_ms"] == 1000
    assert client.usage()["cached"] == 1


def test_the_rpm_setting_drives_the_limiter(tmp_path):
    paced = LLM(make_settings(tmp_path, llm_rpm=120.0))
    assert paced.limiter.rate == pytest.approx(2.0)

    off = LLM(make_settings(tmp_path, llm_rpm=0.0))
    assert off.limiter.acquire() == 0.0


# ---- throttling ----------------------------------------------------------- #


def test_retry_after_beats_the_backoff(tmp_path, monkeypatch):
    paused: list[float] = []
    monkeypatch.setattr(llm_module, "_wait", paused.append)
    transport = Recorder(
        LLMResponse(429, {"error": "slow down"}, {"Retry-After": "7"}),
        LLMResponse(200, body("after the wait")),
    )
    client = LLM(make_settings(tmp_path), transport=transport)

    assert client.chat(HELLO).text == "after the wait"
    assert paused == [7.0]
    assert client.usage()["throttled"] == 1
    assert client.usage()["waited_ms"] == 7000


def test_a_retry_after_date_is_understood(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_module, "_wait", lambda s: None)
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=12))
    transport = Recorder(
        LLMResponse(429, {"error": "slow down"}, {"retry-after": when}),
        LLMResponse(200, body("ok")),
    )
    client = LLM(make_settings(tmp_path), transport=transport)

    client.chat(HELLO)
    assert 9_000 <= client.usage()["waited_ms"] <= 13_000


def test_a_hostile_retry_after_cannot_hang_the_run(tmp_path, monkeypatch):
    paused: list[float] = []
    monkeypatch.setattr(llm_module, "_wait", paused.append)
    transport = Recorder(
        LLMResponse(429, {"error": "go away"}, {"Retry-After": "86400"}),
        LLMResponse(200, body("ok")),
    )
    client = LLM(make_settings(tmp_path), transport=transport)

    client.chat(HELLO)
    assert paused == [llm_module.MAX_RETRY_AFTER]


def test_a_429_without_a_header_falls_back_to_the_backoff(tmp_path):
    transport = Recorder(LLMResponse(429, {"error": "slow down"}), LLMResponse(200, body("ok")))
    client = LLM(make_settings(tmp_path), transport=transport)

    assert client.chat(HELLO).text == "ok"
    assert client.usage()["throttled"] == 1


def test_usage_counts_calls_hits_and_tokens(tmp_path):
    transport = Recorder(LLMResponse(200, body("a", prompt=11, completion=7)))
    client = LLM(make_settings(tmp_path), transport=transport)

    client.chat(HELLO)
    client.chat([{"role": "user", "content": "different"}])
    client.chat(HELLO)  # served from disk

    assert client.usage() == {
        "calls": 3,
        "cached": 1,
        "prompt_tokens": 22,
        "completion_tokens": 14,
        "throttled": 0,
        "waited_ms": 0,
    }
