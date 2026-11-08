"""Load config/*.yaml into typed models. scoring.learned.yaml (if present) overrides scoring.yaml."""
from pathlib import Path
import yaml
from pydantic import BaseModel


class StageCfg(BaseModel):
    models: list[str]
    temperature: float = 0.2


class ModelsConfig(BaseModel):
    base_url: str
    api_key_env: str = "NVIDIA_API_KEY"
    rpm: int = 40
    concurrency: int = 8
    timeout_s: float = 120
    max_retries: int = 3
    backoff_base_s: float = 2.0
    rate_limit_retries: int = 10          # 429: wait and retry the same model (never downgrade)
    rate_limit_backoff_max_s: float = 60.0
    max_tokens: int = 2048
    votes: int = 1
    vote_temperature: float = 0.7
    stages: dict[str, StageCfg]
    system_prefix: dict[str, str] = {}


class WindowCfg(BaseModel):
    min_s: float; max_s: float; sweet_min_s: float; sweet_max_s: float; hook_lead_max_s: float


class WindowWeights(BaseModel):
    hook: float; cliff: float; rules: float; length_fit: float


class Penalties(BaseModel):
    mid_line_cut: float; scene_jump: float; camera_heavy: float; camera_heavy_ratio: float


class FinalWeights(BaseModel):
    hook: float; cliff: float; standalone: float; story_importance: float; rules: float


class Flags(BaseModel):
    spoiler_major: float; resolution_leak: float; hook3s_low: float; talky: float = 0.75


class EarlyEp(BaseModel):
    max_ep: int; bonus: float


class Diversity(BaseModel):
    top_k: int; max_per_scene: int; max_per_act: int


class PairwiseCfg(BaseModel):
    rounds: int; k_factor: float; start_rating: float; seed: int


class RefineCfg(BaseModel):
    max_options: int; max_shift_shots: int


class StitchCfg(BaseModel):
    top_hooks: int; top_cliffs: int; max_combos: int; min_segment_s: float


class FeedbackCfg(BaseModel):
    min_marked: int; max_nudge: float; max_pos: int; max_neg: int


class ScoringConfig(BaseModel):
    hook_min: float
    window: WindowCfg
    window_weights: WindowWeights
    penalties: Penalties
    nms_overlap: float
    k_per_episode: int
    judge_top_n: int
    finalists_n: int
    final_weights: FinalWeights
    flags: Flags
    hook3s_min: float
    early_episode: EarlyEp
    diversity: Diversity
    quote_min_ratio: float
    parse_fail_max_ratio: float
    pairwise: PairwiseCfg
    refine: RefineCfg
    stitch: StitchCfg
    feedback: FeedbackCfg


class SignalsConfig(BaseModel):
    keywords: dict[str, float]
    emotion_intensity: dict[str, float]
    sfx_cues: dict[str, float]
    position: dict[str, float]
    density: dict[str, float]
    visual_terms: list[str]


class AppConfig(BaseModel):
    models: ModelsConfig
    scoring: ScoringConfig
    signals: SignalsConfig


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_config(config_dir: Path) -> AppConfig:
    config_dir = Path(config_dir)
    scoring = _load(config_dir / "scoring.yaml")
    learned = config_dir / "scoring.learned.yaml"
    if learned.exists():
        scoring = _deep_merge(scoring, _load(learned))
    return AppConfig(models=ModelsConfig(**_load(config_dir / "models.yaml")),
                     scoring=ScoringConfig(**scoring),
                     signals=SignalsConfig(**_load(config_dir / "signals.yaml")))
