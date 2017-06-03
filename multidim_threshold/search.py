from itertools import repeat
from collections import deque
from math import isclose

import numpy as np
import funcy as fn

from multidim_threshold.utils import Result, Rec, to_rec


def binsearch(r: Rec, oracle, eps=1e-3):
    """Binary search over the diagonal of the rectangle.

    Returns the lower and upper approximation on the diagonal.
    """
    lo, hi = 0, 1
    bot, top = map(np.array, r)
    diag = top - bot
    f = lambda t: bot + t * diag
    feval = lambda t: oracle(f(t))
    polarity = not feval(lo)

    # Early termination via bounds checks
    if polarity and feval(lo):
        return f(lo), None, f(lo)
    elif not polarity and feval(hi):
        return f(hi), None, f(hi)

    mid = lo
    while (f(hi) - f(lo) > eps).any():
        mid = lo + (hi - lo) / 2
        lo, hi = (mid, hi) if feval(mid) ^ polarity else (lo, mid)

    return f(lo), f(mid), f(hi)


def weightedbinsearch(r: Rec, oracle, eps=0.01):
    lo, hi = 0, 1
    bot, top = map(np.array, r)
    diag = top - bot
    f = lambda t: bot + t * diag
    frobust = lambda t: oracle(f(t))
    # They are opposite signed
    frhi, frlo = frobust(hi), frobust(lo)
    polarity = np.sign(frlo)

    # Early termination via bounds checks
    if frhi * frlo >= 0:
        flo, fhi = f(lo), f(hi)
        fmid = flo if frhi < 0 else fhi
        return flo, None, fhi

    mid = lo
    while (f(hi) - f(lo) > eps).any():
        ratio = frlo / (frhi - frlo)
        mid = lo - (hi - lo) * ratio
        frmid = frobust(mid)

        # Check if we've almost crossed the boundary
        # Note: Because diag is opposite direction of
        # the boundary, the crossing point is unique.
        if isclose(frmid, 0, abs_tol=eps):
            lo, hi = mid - eps / 2, mid + eps / 2
            break

        lo, hi = (mid, hi) if frmid * frhi < 0 else (lo, mid)
        frlo, frhi = frobust(lo), frobust(hi)

    return f(lo), f(mid), f(hi)


def basis_vec(i, dim):
    """Basis vector i"""
    a = np.zeros(dim)
    a[i] = 1.0
    return a


@fn.memoize
def basis_vecs(dim):
    """Standard orthonormal basis."""
    return [basis_vec(i, dim) for i in range(dim)]



def gridSearch(lo, hi, oracle, eps=0.1):
    r = to_rec(lo, hi)
    dim = len(r.bot)
    basis = basis_vecs(dim)
    polarity = not oracle(r.bot)
    queue, mids = deque([(r.bot, None)]), set()
    children = lambda node: (eps * b + node for b in basis)
    while queue:
        node, prev = queue.pop()
        if not(oracle(node) ^ polarity):
            mid = eps / 2 * (prev - node) + node
            mids.add(tuple(list(mid)))
        else:
            queue.extendleft(zip(children(node), repeat(node)))

    return Result(unexplored=[])
