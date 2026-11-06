import shutil
from pathlib import Path
from promofinder.config import load_config
from promofinder.models import HookTag


def test_loads_defaults():
    cfg = load_config(Path("config"))
    assert cfg.scoring.window.max_s == 60.0
    assert cfg.models.stages["tagger"].models[0].startswith("nvidia/")


def test_learned_overrides(tmp_path):
    shutil.copytree("config", tmp_path / "c")
    (tmp_path / "c" / "scoring.learned.yaml").write_text(
        "final_weights: {hook: 0.35, cliff: 0.30, standalone: 0.20, story_importance: 0.05, rules: 0.10}\n")
    assert load_config(tmp_path / "c").scoring.final_weights.hook == 0.35


def test_score_coercion():
    t = HookTag(beat_id="E01-B003", hook_score="8/10", story_importance=11, kind="slap", why="x", quote="x", extra="ignored")
    assert (t.hook_score, t.story_importance) == (8.0, 10.0)
