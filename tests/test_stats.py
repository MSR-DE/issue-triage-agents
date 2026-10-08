"""Statistics used in every scorer."""
import pytest

from evals.stats import ci, mcnemar, wilson


def test_wilson_known_values():
    lo, hi = wilson(15, 30)
    assert (round(lo, 3), round(hi, 3)) == (0.332, 0.668)
    assert wilson(0, 30)[0] == 0.0 and wilson(0, 30)[1] == pytest.approx(0.114, abs=1e-3)
    assert wilson(30, 30)[1] == 1.0 and wilson(30, 30)[0] == pytest.approx(0.886, abs=1e-3)


def test_ci_text():
    assert ci(15, 30) == "15/30 = 50% (95% CI 33%-67%)"
    assert ci(0, 0) == "0/0"


def test_mcnemar():
    assert mcnemar([], []) == 1.0
    assert mcnemar(list(range(11)), []) == pytest.approx(0.0009765625)   # fixed 11 / broke 0
    assert mcnemar([1, 2, 3, 4], []) == pytest.approx(0.125)
    assert mcnemar([1], [2]) == 1.0
