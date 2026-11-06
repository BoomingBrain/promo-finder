from promofinder.prompts import render_anchors


def test_render_anchors_spans_scale():
    a = render_anchors()
    assert "(score 3/10)" in a and "(score 9/10)" in a
    assert a.count("EXAMPLE") >= 6
    assert "HOOK" in a and "CLIFFHANGER" in a


def test_anchors_carry_distribution_rule():
    assert "at most 2" in render_anchors()
