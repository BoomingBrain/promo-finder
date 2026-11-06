from promofinder.models import EpisodeTags, HookTag, CliffTag
from promofinder.voting import merge_votes


def H(b, s):
    return HookTag(beat_id=b, hook_score=s, story_importance=5, kind="slap", why="w", quote="q")


def C(b, s, lvl):
    return CliffTag(beat_id=b, cliff_score=s, story_importance=5, open_question="?", resolved_by=None, spoiler_level=lvl, quote="q")


def test_majority_and_average():
    runs = [EpisodeTags(ep=1, hooks=[H("E01-B003", 9), H("E01-B004", 6)], cliffhangers=[], context_beats=[]),
            EpisodeTags(ep=1, hooks=[H("E01-B003", 7)], cliffhangers=[], context_beats=[]),
            EpisodeTags(ep=1, hooks=[H("E01-B003", 8), H("E01-B010", 5)], cliffhangers=[], context_beats=[])]
    m = merge_votes(runs, 3)
    assert [h.beat_id for h in m.hooks] == ["E01-B003"] and m.hooks[0].hook_score == 8.0
    assert m.votes == 3 and abs(m.agreement - 1 / 3) < 1e-9


def test_categorical_mode():
    runs = [EpisodeTags(ep=1, hooks=[], cliffhangers=[C("E01-B012", 9, "major")], context_beats=["E01-B004"]),
            EpisodeTags(ep=1, hooks=[], cliffhangers=[C("E01-B012", 7, "none")], context_beats=["E01-B004"]),
            EpisodeTags(ep=1, hooks=[], cliffhangers=[C("E01-B012", 8, "none")], context_beats=[])]
    m = merge_votes(runs, 3)
    assert m.cliffhangers[0].spoiler_level == "none" and m.cliffhangers[0].cliff_score == 8.0
    assert m.context_beats == ["E01-B004"]


def test_single_vote_passthrough():
    r = EpisodeTags(ep=1, hooks=[H("E01-B003", 9)], cliffhangers=[], context_beats=[])
    assert merge_votes([r], 1).agreement is None
