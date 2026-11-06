from pathlib import Path
from promofinder.bible import build_bible, render_context
from promofinder.config import load_config
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Act, Bible, Episode, Reveal
from tests.fakes import FakeLLM

B = Bible(premise="A jilted bride marries a wounded soldier.", characters={"Mara": "jilted bride, the heroine"},
          arc=[Act(name="Betrayal", ep_start=1, ep_end=5, summary="s")], episode_summaries={1: "accused", 2: "meets stranger"},
          major_reveals=[Reveal(desc="soldier is a duke", ep=30, setup_eps=[2, 9])])


def test_context_layers():
    c = render_context(B, 1)
    assert "Mara — jilted bride" in c and "Betrayal (1–5)" in c and "soldier is a duke (ep 30)" in c
    assert "previous episode: none" in c and "next episode: meets stranger" in c
    assert render_context(None, 1) == ""


def _ep(n):
    return Episode(ep=n, logline=f"logline {n}", continuity_notes=f"notes {n}", duration_s=60, scenes=[], beats=[],
                   source_file=f"e{n}.txt")


def test_build_bible_sends_all_loglines_and_converts_keys():
    reply = {"premise": "p", "characters": {"Mara": "heroine"}, "arc": [{"name": "A", "ep_start": 1, "ep_end": 2}],
             "episode_summaries": {"1": "one", "2": "two"}, "major_reveals": []}
    llm = FakeLLM([reply])
    b = build_bible([_ep(1), _ep(2)], llm, load_config(Path("config")), RunMeta())
    assert b.episode_summaries == {1: "one", 2: "two"}
    stage, messages, _, _ = llm.seen[0]
    text = " ".join(m["content"] for m in messages)
    assert stage == "bible" and "Episode 1: logline 1 / notes 1" in text and "Episode 2: logline 2 / notes 2" in text


def test_build_bible_returns_none_on_failure():
    meta = RunMeta()
    assert build_bible([_ep(1)], FakeLLM([LLMStageError("boom")]), load_config(Path("config")), meta) is None
    assert any("bible" in w for w in meta.warnings)
