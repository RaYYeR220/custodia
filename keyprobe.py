import pathlib
from custodia.demo import load_corpus, to_turns
from custodia.prompts import build_extract_messages
from custodia.llm import cache_key
from custodia.config import settings
from custodia.extract import EXTRACT_MAX_TOKENS

s = settings()
d = load_corpus()
turns = to_turns(d["sessions"][0], "demo", 0)
msgs = build_extract_messages(turns, list(range(len(turns))), d.get("principal", "user"))
k = cache_key(s.extract_model, msgs, 0.0, EXTRACT_MAX_TOKENS)
seed = pathlib.Path(s.cache_seed_dirs[0]) / "llm" / k[:2] / f"{k}.json"
print("model", s.extract_model, "max_tokens", EXTRACT_MAX_TOKENS)
print("key", k[:16], "exists in seed cache:", seed.exists())
print("prompt chars", sum(len(m["content"]) for m in msgs))
