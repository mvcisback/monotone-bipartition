"""Implements muli-dimensional threshold discovery via binary search."""
from collections import namedtuple, Iterable
from itertools import combinations
from heapq import heappush as hpush, heappop as hpop
from math import isclose

import numpy as np
from numpy import array
import funcy as fn

Rec = namedtuple("Rec", "bot top")
Result = namedtuple("Result", "vol mids unexplored")


def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def binsearch(r: Rec, stleval, eps=1e-3):
    """Binary search over the diagonal of the rectangle.

    Returns the lower and upper approximation on the diagonal.
    """
    lo, hi = 0, 1
    diag = r.top - r.bot
    f = lambda t: r.bot + t * diag
    feval = lambda t: stleval(f(t))
    polarity = not feval(lo)

    # Early termination via bounds checks
    if polarity and feval(lo):
        return f(lo), f(lo), f(lo)
    elif not polarity and feval(hi):
        return f(hi), f(hi), f(hi)

    while (f(hi) - f(lo) > eps).any():
        mid = lo + (hi - lo) / 2
        lo, hi = (mid, hi) if feval(mid) ^ polarity else (lo, mid)

    return f(lo), f(mid), f(hi)


def weightedbinsearch(r: Rec, robust, eps=0.01) -> (array, array, array):
    lo, hi = 0, 1
    diag = r.top - r.bot
    f = lambda t: r.bot + t * diag
    frobust = lambda t: robust(f(t))
    # They are opposite signed
    frhi, frlo = frobust(hi), frobust(lo)
    polarity = np.sign(frlo)

    # Early termination via bounds checks
    if frhi * frlo >= 0:
        flo, fhi = f(lo), f(hi)
        fmid = flo if frhi < 0 else fhi
        return flo, fmid, fhi

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


def gridSearch(lo, hi, is_member, eps=0.1):
    r = to_rec(lo, hi)
    dim = len(r.bot)
    basis = basis_vecs(dim)
    polarity = not is_member(r.bot)
    queue, mids = [(r.bot, None)], set()
    children = lambda node: (eps * b + node for b in basis)
    while queue:
        node, prev = hpop(queue)
        if not(is_member(node) ^ polarity):
            mid = eps / 2 * (prev - node) + node
            mids.add(tuple(list(mid)))
        else:
            for c in children(node):
                hpush(queue, (c, node))

    return Result(vol=eps**dim * len(mids), mids=mids, unexplored=[])


def to_tuple(r: Rec):
    return tuple(map(tuple, r))


def forward_cone(p: array, r: Rec) -> Rec:
    """Computes the forward cone from point p."""
    return Rec(p, r.top)


def backward_cone(p: array, r: Rec) -> Rec:
    """Computes the backward cone from point p."""
    return Rec(r.bot, p)


def basis_vec(i, dim):
    """Basis vector i"""
    a = np.zeros(dim)
    a[i] = 1.0
    return a


@fn.memoize
def basis_vecs(dim):
    """Standard orthonormal basis."""
    return [basis_vec(i, dim) for i in range(dim)]


def generate_incomparables(mid, r):
    """Generate all incomparable hyper-boxes."""
    forward, backward = forward_cone(mid, r), backward_cone(mid, r)
    bases = (backward.bot, forward.bot)
    diags = (backward.top - backward.bot, forward.top - forward.bot)
    dim = len(bases[0])
    basis = basis_vecs(dim)
    for i in range(1, dim):
        for es in combinations(basis, i):
            vs = tuple(sum((diag @e) * e for e in es) for diag in diags)
            yield Rec(*[base + v for base, v in zip(bases, vs)])


def subdivide(low, mid, high, r: Rec) -> [Rec]:
    """Computes the set of incomparable cones of point p."""
    forward = forward_cone(high, r)
    backward = backward_cone(low, r)
    incomparables = list(generate_incomparables(mid, r))
    return backward, forward, incomparables


def volume(rec: Rec):
    return np.prod(np.abs(rec.bot - rec.top))


def multidim_search(lo, hi, is_member, diagsearch=None):
    """Generator for iteratively approximating the oracle's threshold."""
    rec = to_rec(lo, hi)

    if diagsearch is None:
        bool_oracle = isinstance(is_member(rec.bot), bool)
        diagsearch = binsearch if bool_oracle else weightedbinsearch

    initial_vol = unknown_vol = volume(rec)
    queue = [(unknown_vol, rec)]
    mids = set()
    while queue:
        _, rec = hpop(queue)
        rec = Rec(*map(np.array, rec))
        low, mid, high = diagsearch(rec, is_member)
        backward, forward, incomparables = subdivide(low, mid, high, rec)
        mids.add(tuple(list(mid)))

        for r in incomparables:
            hpush(queue, (-volume(r), to_tuple(r)))

        # not correct, since is doesn't include upward closure's area
        unknown_vol -= volume(backward) + volume(forward)
        est_pct_vol = unknown_vol / initial_vol
        yield Result(est_pct_vol, mids, queue)
