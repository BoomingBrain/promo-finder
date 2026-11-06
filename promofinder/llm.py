"""One OpenAI-compatible client for every stage: per-stage fallback chains, retries with backoff,
rate limiting, an on-disk cache (keyed by model+messages+temperature+vote_index) and JSON extraction."""
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from pydantic import BaseModel, ValidationError
from promofinder.config import ModelsConfig
from promofinder.meta import RunMeta


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class TransientError(Exception):
    """429 / 5xx / timeout: worth retrying."""


class RateLimited(TransientError):
    """429: the model is fine but busy. Wait and retry the SAME model; never downgrade because of it."""


class ModelNotFound(Exception):
    """Model id unknown or not available to this key."""


class ModelRejected(Exception):
    """Any other 4xx (bad request, auth, forbidden): this model can't answer this call; try the next one."""


class LLMStageError(Exception):
    """Every model in a stage's chain failed."""


Transport = Callable[[str, list, float, int], Completion]


def openai_transport(base_url: str, api_key: str, timeout_s: float) -> Transport:
    import openai
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    def call(model: str, messages: list, temperature: float, max_tokens: int) -> Completion:
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                               max_tokens=max_tokens)
        except openai.RateLimitError as e:
            raise RateLimited(str(e)) from e
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
            raise TransientError(str(e)) from e
        except openai.NotFoundError as e:
            raise ModelNotFound(f"{model}: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code in (404, 410, 422):
                raise ModelNotFound(f"{model}: {e}") from e
            if e.status_code == 429:
                raise RateLimited(str(e)) from e
            if e.status_code >= 500:
                raise TransientError(str(e)) from e
            raise ModelRejected(f"{model}: HTTP {e.status_code}: {str(e)[:300]}") from e
        msg = r.choices[0].message if r.choices else None
        text = (msg.content or "") if msg else ""
        if not text.strip() and msg is not None:
            # Reasoning models sometimes leave `content` empty and put everything in a reasoning field.
            extra = msg.model_extra or {}
            text = extra.get("reasoning_content") or extra.get("reasoning") or ""
        u = r.usage
        return Completion(text, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
    return call


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?", re.I)


def extract_json(text: str):
    """Strip <think> blocks and code fences, then return the first balanced {...} or [...] that parses."""
    t = _FENCE.sub("", _THINK.sub("", text or ""))
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model reply")


class RateLimiter:
    def __init__(self, rpm: int, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / max(1, rpm)
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = None

    def acquire(self) -> None:
        with self._lock:
            now = self.clock()
            if self._next is None or now >= self._next:
                self._next = now + self.interval
                return
            wait = self._next - now
            self._next += self.interval
        self.sleep(wait)


class LLMClient:
    def __init__(self, cfg: ModelsConfig, cache_dir: Path, meta: RunMeta, transport: Transport | None = None,
                 sleep=time.sleep):
        self.cfg, self.meta, self.sleep = cfg, meta, sleep
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        if transport is None:
            key = os.environ.get(cfg.api_key_env, "")
            if not key:
                raise RuntimeError(f"Missing {cfg.api_key_env} (see .env.example)")
            transport = openai_transport(cfg.base_url, key, cfg.timeout_s)
        self.transport = transport
        self.limiter = RateLimiter(cfg.rpm, sleep=sleep)

    def _key(self, model, msgs, temperature, vote_index) -> Path:
        h = hashlib.sha256(json.dumps([model, msgs, temperature, vote_index], ensure_ascii=False).encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    @staticmethod
    def _read_cache(path: Path) -> Completion | None:
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return Completion(d["text"], d["prompt_tokens"], d["completion_tokens"])
        except (ValueError, KeyError, TypeError):
            return None                                   # truncated/corrupt entry: treat as a miss and overwrite

    def _call(self, stage, model, msgs, temperature, fallback) -> Completion:
        """Retries: rate limits wait on the same model (capped exponential backoff, up to rate_limit_retries);
        other transient errors retry max_retries times. Raises when exhausted."""
        k = rl = 0
        while True:
            self.limiter.acquire()
            t = time.monotonic()
            try:
                c = self.transport(model, msgs, temperature, self.cfg.max_tokens)
                self.meta.record_call(stage, model, c.prompt_tokens, c.completion_tokens, time.monotonic() - t, False, fallback)
                return c
            except RateLimited:
                if rl >= self.cfg.rate_limit_retries:
                    raise
                self.meta.incr("rate_limit_waits")
                self.sleep(min(self.cfg.rate_limit_backoff_max_s, self.cfg.backoff_base_s * 2 ** rl))
                rl += 1
            except TransientError:
                if k >= self.cfg.max_retries:
                    raise
                self.sleep(self.cfg.backoff_base_s * 2 ** k)
                k += 1

    def _run(self, stage, messages, parse, vote_index, temperature):
        st = self.cfg.stages[stage]
        temp = st.temperature if temperature is None else temperature
        for i, model in enumerate(st.models):
            if i > 0:
                self.meta.incr("fallback_switches")
            prefix = self.cfg.system_prefix.get(model)
            msgs = ([{"role": "system", "content": prefix}] if prefix else []) + list(messages)
            try:
                for attempt in (1, 2):
                    path = self._key(model, msgs, temp, vote_index)
                    comp = self._read_cache(path)
                    if comp is not None:
                        self.meta.record_call(stage, model, comp.prompt_tokens, comp.completion_tokens, 0.0, True, i > 0)
                    else:
                        comp = self._call(stage, model, msgs, temp, i > 0)
                    try:
                        result = parse(comp.text)
                    except (ValueError, ValidationError) as e:
                        self.meta.incr("json_retries")
                        msgs = msgs + [{"role": "assistant", "content": comp.text[:4000]},
                                       {"role": "user", "content": f"Your reply was invalid: {str(e)[:500]}. "
                                        "Reply with only valid JSON matching the schema."}]
                        continue
                    if self._read_cache(path) is None:
                        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
                        tmp.write_text(json.dumps({"text": comp.text, "prompt_tokens": comp.prompt_tokens,
                                                   "completion_tokens": comp.completion_tokens}, ensure_ascii=False),
                                       encoding="utf-8")
                        os.replace(tmp, path)                     # atomic: a killed run never leaves half a file
                    return result
            except RateLimited as e:
                # a busy model is not a broken one: fail this call loudly rather than silently use a weaker model
                self.meta.incr("rate_limit_exhausted")
                self.meta.warn(f"{stage}: model {model} still rate-limited after {self.cfg.rate_limit_retries} waits; "
                               "call skipped (rerun to retry) instead of downgrading")
                raise LLMStageError(f"stage {stage}: {model} rate-limited") from e
            except (TransientError, ModelNotFound, ModelRejected) as e:
                self.meta.warn(f"{stage}: model {model} failed ({type(e).__name__}: {str(e)[:200]}); trying next")
                continue
        raise LLMStageError(f"stage {stage}: every model in the chain failed")

    def complete_json(self, stage: str, messages: list, schema: type[BaseModel], vote_index: int = 0,
                      temperature: float | None = None) -> BaseModel:
        return self._run(stage, messages, lambda txt: schema.model_validate(extract_json(txt)), vote_index, temperature)

    def complete_text(self, stage: str, messages: list, vote_index: int = 0, temperature: float | None = None) -> str:
        def parse(txt):
            t = _THINK.sub("", txt or "").strip()
            if not t:
                raise ValueError("empty reply")
            return t
        return self._run(stage, messages, parse, vote_index, temperature)
