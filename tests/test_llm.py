import pytest
from pathlib import Path
from pydantic import BaseModel
from promofinder.config import load_config
from promofinder.llm import LLMClient, Completion, TransientError, ModelNotFound, LLMStageError, extract_json, RateLimiter
from promofinder.meta import RunMeta


class Out(BaseModel):
    x: int


def fake(script):
    calls = []

    def t(model, messages, temperature, max_tokens):
        calls.append(model)
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return Completion(item, 10, 5)
    t.calls = calls
    return t


def client(tmp_path, script):
    m = RunMeta(); t = fake(script)
    return LLMClient(load_config(Path("config")).models, tmp_path, m, transport=t, sleep=lambda s: None), m, t


Q = [{"role": "user", "content": "q"}]


def test_extract_json_variants():
    assert extract_json('<think>hmm {"no": 1}</think>```json\n{"x": 1}\n```') == {"x": 1}
    assert extract_json('Sure! [{"a":2}] done') == [{"a": 2}]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_retry_then_success(tmp_path):
    c, m, t = client(tmp_path, [TransientError(), '{"x": 3}'])
    assert c.complete_json("judge", Q, Out).x == 3


def test_invalid_json_twice_falls_back(tmp_path):
    c, m, t = client(tmp_path, ["nope", "still nope", '{"x": 4}'])
    assert c.complete_json("judge", Q, Out).x == 4
    assert t.calls[2] != t.calls[0] and m.counters["fallback_switches"] == 1 and m.counters["json_retries"] == 2


def test_model_not_found_falls_back(tmp_path):
    c, m, t = client(tmp_path, [ModelNotFound(), '{"x": 5}'])
    assert c.complete_json("judge", Q, Out).x == 5


def test_all_fail_raises(tmp_path):
    c, m, t = client(tmp_path, [ModelNotFound(), ModelNotFound()])
    with pytest.raises(LLMStageError):
        c.complete_json("judge", Q, Out)


def test_cache_hit_and_vote_index_in_key(tmp_path):
    c, m, t = client(tmp_path, ['{"x": 1}', '{"x": 2}'])
    assert c.complete_json("judge", Q, Out).x == 1
    assert c.complete_json("judge", Q, Out).x == 1          # cached, no transport call
    assert c.complete_json("judge", Q, Out, vote_index=1).x == 2
    assert len(t.calls) == 2 and sum(1 for k in m.calls if k["cached"]) == 1


def test_rate_limiter_spacing():
    now = [0.0]; slept = []
    rl = RateLimiter(60, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    rl.acquire(); rl.acquire()
    assert slept and abs(slept[0] - 1.0) < 1e-6
