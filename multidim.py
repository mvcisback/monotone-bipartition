"""Implements muli-dimensional threshold discovery via binary search."""
from collections import namedtuple, deque

from numpy import array

Rec = namedtuple("Rec", "bot top")


def binsearch(r: Rec, is_member, eps=0.1) -> (array, array, array):
    """Binary search over the diagonal of the rectangle.
    
    Returns the lower and upper approximation on the diagonal.
    """
    lo, hi = 0, 1
    diag = r.top - r.bot
    f = lambda t: r.bot + t * diag
    while hi - lo > eps:
        mid = lo + (hi - lo) / 2
        if not is_member(f(mid)):
            lo, hi = mid, hi
        else:
            hi, lo = lo, mid
    return f(lo), f(mid), f(hi)


def forward_cone(p: array, r: Rec) -> Rec:
    """Computes the forward cone from point p."""
    return Rec(p, r.top)


def backward_cone(p: array, r: Rec) -> Rec:
    """Computes the backward cone from point p."""
    return Rec(r.bot, p)


def incomparable(p: array, r: Rec) -> [Rec]:
    """Computes the set of incomparable cones of point p."""
    r01 = Rec(array(r.bot[0], p[1]), array(p[0], r.top[1]))
    r10 = Rec(array(p[0], r.bot[1]), array(r.top[0], p[1]))
    return [r01, r10]


def multidim_search(rec: Rec, is_member) -> [(set(Rec), set(Rec))]:
    """Generator for iteratively approximating the oracle's threshold."""
    queue = deque([rec])
    good_approx, bad_approx = set(), set()
    while True:
        rec = queue.pop()
        low, mid, high = binsearch(rec, is_member)

        bad_approx.add(backward_cone(low, rec))
        good_approx.add(forward_cone(high, rec))
        queue.extendleft(incomparable(mid, rec))

        yield bad_approx, good_approx
