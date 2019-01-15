from enum import Enum, auto

import funcy as fn
import numpy as np

from monotone_bipartition import rectangles as mdtr

EPS = 1e-2


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
