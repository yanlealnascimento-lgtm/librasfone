import pytest

from medicao import _percentile, _stats


def test_percentile_endpoints_and_median():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 5.0
    assert _percentile(values, 50) == pytest.approx(3.0)


def test_percentile_interpolates_between_points():
    # p95 de 0..100 (101 valores) cai em 95.
    values = [float(i) for i in range(101)]
    assert _percentile(values, 95) == pytest.approx(95.0)


def test_stats_handles_empty_input():
    assert _stats([]) == "sem dados"


def test_stats_reports_count_and_summary():
    text = _stats([10.0, 20.0, 30.0])
    assert "n=3" in text
    assert "média=20" in text
