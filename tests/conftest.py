"""Test-session guards.

Two things must be true for every test run, on every machine, whether or not a
developer has credentials configured locally:

* **no test reaches a provider.** A real key in `.env` must not turn the suite
  into a bill, and a test that silently depends on a live model is not a test.
  The environment is neutralised before `custodia.config` is first read.
* **live-graph tests are opt-out.** They are marked `graph` and skipped
  automatically when HydraDB is not listening, so the suite still runs on a
  machine with nothing started.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

os.environ["CUSTODIA_LLM_API_KEY"] = ""
os.environ["CUSTODIA_CACHE_ONLY"] = "true"
os.environ.setdefault("CUSTODIA_CACHE_DIR", os.path.join(os.path.dirname(__file__), "_cache"))


def _graph_is_up() -> bool:
    uri = os.environ.get("HYDRA_URI", "neo4j://127.0.0.1:7687")
    parsed = urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


GRAPH_UP = _graph_is_up()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if GRAPH_UP:
        return
    skip = pytest.mark.skip(reason="HydraDB is not listening; start it with docker compose up -d hydradb")
    for item in items:
        if "graph" in item.keywords:
            item.add_marker(skip)
