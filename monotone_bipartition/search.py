from enum import Enum, auto

import funcy as fn
import numpy as np

from monotone_bipartition import rectangles as mdtr
from monotone_bipartition import refine

EPS = 1e-4


class SearchResultType(Enum):
    TRIVIALLY_FALSE = auto()
    TRIVIALLY_TRUE = auto()
    NON_TRIVIAL = auto()


def diagonal_convex_comb(r):
    bot, top = np.array(r.bot), np.array(r.top)
    diag = top - bot
    return lambda t: bot + t * diag


def binsearch(r, oracle, eps=EPS, find_lambda=False):
    """Binary search over the diagonal of the rectangle.

    Returns the lower and upper approximation on the diagonal.
    """
    f = diagonal_convex_comb(r)
    feval = fn.compose(oracle, f)
    lo, hi = 0, 1

    # Early termination via bounds checks
    if feval(lo):
        result_type = SearchResultType.TRIVIALLY_TRUE
        hi = 0
    elif not feval(hi):
        result_type = SearchResultType.TRIVIALLY_FALSE
    else:
        result_type = SearchResultType.NON_TRIVIAL
        mid = lo
        while hi - lo > eps:
            mid = lo + (hi - lo) / 2
            lo, hi = (lo, mid) if feval(mid) else (mid, hi)

    if find_lambda:
        if result_type == SearchResultType.TRIVIALLY_TRUE:
            return result_type, -1
        elif result_type == SearchResultType.TRIVIALLY_FALSE:
            return result_type, 2
        return result_type, (lo+hi)/2
    else:
        return result_type, mdtr.to_rec(zip(f(lo), f(hi)))


def line_intersect(func, point, tol, *, percent=False):
    box_intersect = np.array(point) / max(point)
    origin = [0]*len(point)
    rec = mdtr.to_rec(zip(origin, box_intersect))  # Compute bounding rec.
    return binsearch(rec, func, eps=tol, find_lambda=percent)[1]


def lexicographic_opt(func, ordering, tol):
    dim = len(ordering)
    assert set(fn.pluck(0, ordering)) == set(range(dim))
    tol /= dim  # Need to compensate for multiple binsearches.

    rec = refine.bounding_box(
        domain=mdtr.unit_rec(dim),
        oracle=func
    )
    # If polarity is True, set initial value at bounding.top.
    # O.w. use bounding.bot.
    base = tuple((rec.top if p else rec.bot)[i] for i, p in sorted(ordering))

    res_rec = mdtr.to_rec(zip(base, base))
    for idx, polarity in ordering:
        oracle = func
        rec = mdtr.to_rec(
            (0, 1) if i == idx else (p, p) for i, p in enumerate(base)
        )
        result_type, res_cand = binsearch(rec, oracle, eps=tol)

        if result_type == SearchResultType.NON_TRIVIAL:
            res_rec = res_cand
            base = res_rec.bot

    return res_rec
