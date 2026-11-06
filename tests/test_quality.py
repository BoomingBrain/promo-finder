from pathlib import Path
from promofinder.meta import RunMeta
from promofinder.models import Judgement
from promofinder.quality import risk_notes, run_notes
from tests.test_report import EPS, P


def J(cid, standalone=8.0, **kw):
    return Judgement(cand_id=cid, standalone=standalone, confusing_bits=kw.pop("bits", []), hook_text="h", cliffhanger_text="c",
                     why_it_works="w", spoiler_risk=False, resolution_leak=kw.pop("leak", False), **kw)


def test_clean_promo_has_no_risks():
    p = P(4.0, 30.0, hook="E01-B004"); p.judgement = J(p.candidate.cand_id)
    assert risk_notes(p, [], EPS) == []


def test_flags_spec_failure_categories():
    p = P(4.0, 30.0); p.judgement = J(p.candidate.cand_id, standalone=4.0, bits=["who is Victor?"], leak=True)
    notes = " | ".join(risk_notes(p, [], EPS))
    assert "may need context" in notes and "who is Victor?" in notes
    assert "may resolve itself" in notes
    assert "visual-only hook" in notes            # fixture hook E01-B003 is an ACTION beat (the slap)


def test_near_duplicate_of_higher_rank():
    a = P(4.0, 30.0); a.rank = 1; a.judgement = J(a.candidate.cand_id)
    b = P(9.0, 30.0); b.rank = 2; b.judgement = J(b.candidate.cand_id)
    assert any("near-duplicate of #1" in n for n in risk_notes(b, [a], EPS))


def test_unjudged_and_run_level_notes():
    p = P(4.0, 30.0)
    assert any("not judged" in n for n in risk_notes(p, [], EPS))
    m = RunMeta(); m.set("pairwise_consistent_rate", 0.21); m.incr("skipped_episodes")
    notes = " | ".join(run_notes(m))
    assert "21%" in notes and "skipped" in notes
