"""Thread-safe run metadata: every LLM call, counters, warnings, timings -> run_meta.json."""
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


class RunMeta:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls: list[dict] = []
        self.counters: dict[str, int] = defaultdict(int)
        self.warnings: list[str] = []
        self.values: dict = {}
        self.timings: dict[str, float] = {}

    def record_call(self, stage: str, model: str, prompt_tokens: int, completion_tokens: int,
                    latency_s: float, cached: bool, fallback: bool) -> None:
        with self._lock:
            self.calls.append(dict(stage=stage, model=model, prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens, latency_s=round(latency_s, 3),
                                   cached=cached, fallback=fallback))

    def incr(self, counter: str, n: int = 1) -> None:
        with self._lock:
            self.counters[counter] += n

    def warn(self, msg: str) -> None:
        with self._lock:
            self.warnings.append(msg)

    def set(self, key: str, value) -> None:
        with self._lock:
            self.values[key] = value

    @contextmanager
    def stage_timer(self, stage: str):
        t = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self.timings[stage] = round(self.timings.get(stage, 0.0) + time.monotonic() - t, 3)

    def summary(self) -> dict:
        live = [c for c in self.calls if not c["cached"]]
        by_stage: dict[str, dict] = {}
        for c in self.calls:
            s = by_stage.setdefault(c["stage"], {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "models": {}})
            s["calls"] += 1; s["cached"] += int(c["cached"])
            s["prompt_tokens"] += c["prompt_tokens"]; s["completion_tokens"] += c["completion_tokens"]
            s["models"][c["model"]] = s["models"].get(c["model"], 0) + 1
        return {"total_calls": len(self.calls), "live_calls": len(live),
                "prompt_tokens": sum(c["prompt_tokens"] for c in live),
                "completion_tokens": sum(c["completion_tokens"] for c in live), "by_stage": by_stage}

    def save(self, path: Path) -> None:
        with self._lock:
            data = {"summary": None, "counters": dict(self.counters), "values": self.values,
                    "timings_s": self.timings, "warnings": self.warnings, "calls": self.calls}
        data["summary"] = self.summary()
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
