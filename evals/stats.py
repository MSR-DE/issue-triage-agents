"""Small statistics helpers shared by the scorers (kept free of heavy imports so the
tests and CI can import them without the embedding model or an LLM)."""
from math import comb, sqrt


def mcnemar(fixed, broke):
    """Exact two-sided McNemar p-value from the two lists of disagreements.
    If B were no better than A, each disagreement would be a fair coin flip."""
    n = len(fixed) + len(broke)
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(len(fixed), len(broke)) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    """95% Wilson score interval (lo, hi) for k successes out of n. Unlike p ± 1.96·SE it
    stays inside [0, 1] and is sensible at small n and at 0/n or n/n."""
    if not n:
        return 0.0, 1.0
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, mid - half), min(1.0, mid + half)


def ci(k, n):
    """'k/n = p% (95% CI lo-hi%)' for printing."""
    if not n:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}-{hi:.0%})"
