from itertools import repeat
from collections import deque
from math import isclose
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
import funcy as fn

from multidim_threshold.rectangles import to_rec, Rec


class SearchResultType(Enum):
    TRIVIALLY_FALSE = auto()
    TRIVIALLY_TRUE = auto()
    NON_TRIVIAL = auto()

SearchResult = Tuple[SearchResultType, Optional[Rec]]

def binsearch(r: Rec, oracle, eps=1e-3) -> SearchResult:
    """Binary search over the diagonal of the rectangle.

    Returns the lower and upper approximation on the diagonal.
    """
    lo, hi = 0, 1
    bot, top = np.array(r.bot), np.array(r.top)
    diag = top - bot
    f = lambda t: bot + t * diag
    feval = lambda t: oracle(f(t))

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
