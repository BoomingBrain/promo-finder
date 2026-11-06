from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.signals import compute_signals
from promofinder.meta import RunMeta
from promofinder.llm import LLMStageError
from promofinder.tagger import build_transcript, quote_matches, tag_episode
from tests.fakes import FakeLLM

CFG = load_config(Path("config"))
EP = load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes[0]
SIG = compute_signals(EP, CFG.signals)


def test_transcript_drops_plain_camera_keeps_visual():
    t = build_transcript(EP, SIG, CFG.signals)
    assert "[E01-B001]" not in t and "[E01-B002]" in t
    assert '[E01-B004] 00:06.50–00:08.50 DIALOGUE VICTOR (Furious): "You stole my mother’s ring!"' in t


def test_quote_normalisation():
    assert quote_matches('"I have never seen that ring..."', "I have never seen that ring…", 0.8)
    assert quote_matches("You stole my mother's", "You stole my mother’s ring!", 0.8)
    assert not quote_matches("Someone put this here", "You stole my mother’s ring!", 0.8)


def test_tag_episode_validates():
    reply = {"hooks": [
        {"beat_id": "E01-B003", "hook_score": "9", "story_importance": 5, "kind": "slap", "why": "slap", "quote": "Victor slaps Mara across"},
        {"beat_id": "E01-B099", "hook_score": 8, "story_importance": 5, "kind": "other", "why": "x", "quote": "x"},
        {"beat_id": "E01-B006", "hook_score": 7, "story_importance": 5, "kind": "other", "why": "x", "quote": "Someone put this here"}],
        "cliffhangers": [{"beat_id": "E01-B012", "cliff_score": 9, "story_importance": 6, "open_question": "who framed her?",
                          "resolved_by": "next_episode", "spoiler_level": "none", "quote": "Ominous string swell rises"}],
        "context_beats": ["E01-B004", "E01-B777"]}
    m = RunMeta()
    tags = tag_episode(EP, SIG, FakeLLM([reply]), CFG, m)
    assert [h.beat_id for h in tags.hooks] == ["E01-B003"] and tags.hooks[0].hook_score == 9.0
    assert tags.cliffhangers[0].resolved_by == "next_episode" and tags.context_beats == ["E01-B004"]
    assert m.counters["unknown_ids"] == 2 and m.counters["quote_mismatches"] == 1


def test_stage_error_skips_episode():
    m = RunMeta()
    assert tag_episode(EP, SIG, FakeLLM([LLMStageError("tagger")]), CFG, m) is None
    assert m.counters["skipped_episodes"] == 1
