from random import sample
from collections import namedtuple

from numpy import array

Rec = namedtuple("Rec", "bot top")


def binsearch(P: Rec, oracle, eps=0.1):
    lo, hi = 0, 1
    diag = P.top - P.bot
    f = lambda t: P.bot + t * diag
    while hi - lo > eps:
        mid = lo + (hi - lo) / 2
        cls = oracle(f(mid))
        if mid == -1:
            lo, hi = mid, hi
        else:
            hi, lo = lo, mid
    return f(lo), f(hi)


def forward_cone(a, X: Rec):
    return Rec(a, X.top)


def backward_cone(b, X: Rec):
    return Rec(X.bot, b)


def incomparable(q, X: Rec):
    r01 = Rec(array(X.bot[0], q[1]), array(q[0], X.top[1]))
    r10 = Rec(array(q[0], X.bot[1]), array(X.top[0], q[1]))
    return set(r01, r10)


def search(X: Rec, oracle, N=100):
    L = set(X)
    good_approx, bad_approx = set(), set()
    while True:
        P = sample(L, 1)
        L.discard(P)
        a, b = binsearch(P)
        bad_approx.add(backward_cone(a))
        forward_approx.add(forward_cone(b))
        q = (a + b) / 2
        L |= incomparable(q)
        yield L
