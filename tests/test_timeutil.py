from promofinder.timeutil import parse_ts, format_ts
import pytest


def test_parse_variants():
    assert parse_ts("00:18.00") == 18.0
    assert parse_ts("(02:13.80)") == 133.8
    assert parse_ts(" 1:05 ") == 65.0
    assert parse_ts("01:02:03.50") == 3723.5


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ts("18 seconds")


def test_format_roundtrip():
    assert format_ts(41.2) == "00:41.20"
    assert format_ts(133.8) == "02:13.80"
    assert format_ts(3723.5) == "62:03.50"
