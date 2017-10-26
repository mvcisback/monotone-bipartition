from enum import Enum, auto
from typing import Optional, Tuple

import funcy as fn
import numpy as np

from multidim_threshold.rectangles import Rec, to_rec


class SearchResultType(Enum):
    TRIVIALLY_FALSE = auto()
    TRIVIALLY_TRUE = auto()
    NON_TRIVIAL = auto()


SearchResult = Tuple[SearchResultType, Optional[Rec]]


def diagonal_convex_comb(r: Rec):
    bot, top = np.array(r.bot), np.array(r.top)
    diag = top - bot
    return lambda t: bot + t * diag


def binsearch(r: Rec, oracle, eps=1e-4) -> SearchResult:
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
        while (f(hi) - f(lo) > eps).any():
            mid = lo + (hi - lo) / 2
            lo, hi = (lo, mid) if feval(mid) else (mid, hi)
    return result_type, to_rec(zip(f(lo), f(hi)))
