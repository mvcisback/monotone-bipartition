"""Implements muli-dimensional threshold discovery via binary search."""
from collections import namedtuple, deque
from itertools import chain, combinations
from pathlib import Path
from heapq import heappush as hpush, heappop as hpop

import numpy as np
from numpy import array
import svgwrite
import funcy as fn

Rec = namedtuple("Rec", "bot top")


def binsearch(r: Rec, is_member, eps=0.01) -> (array, array, array):
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
            lo, hi = lo, mid
    return f(lo), f(mid), f(hi)
    
def weightedbinsearch(r: Rec, is_member, eps=0.01) -> (array, array, array):
    lo, hi = 0, 1
    diag = r.top - r.bot
    f = lambda t: r.bot + t * diag
    if is_member(f(hi))*is_member(f(lo)) < 0:    #They are opposite signed
        while hi - lo > eps:
            mid = lo - is_member(f(lo)) * (hi - lo) / (is_member(f(hi)) -  is_member(f(lo)))
            if is_member(f(mid)) * is_member(f(hi)) < 0:    # That is robustness of hi and robustness of mid are opposite signed
                lo, hi = mid, hi
            elif is_member(f(mid)) * is_member(f(lo)) < 0:
                lo, hi = lo, mid
            else:
                return f(lo), f(mid), f(hi)
    else:
        # TO-DO: what happens if the entire space is of same robustness
        return f(lo), f(lo), f(hi)
    return f(lo), f(mid), f(hi)    
                    
            

        


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
            vs = tuple(sum((diag @e)*e for e in es) for diag in diags)
            yield Rec(*[base + v for base, v in zip(bases, vs)])


def subdivide(low, mid, high, r: Rec) -> [Rec]:
    """Computes the set of incomparable cones of point p."""
    forward = forward_cone(high, r)
    backward = backward_cone(low, r)
    incomparables = list(generate_incomparables(mid, r))
    return backward, forward, incomparables


def volume(rec: Rec):
    return np.prod(np.abs(rec.bot-rec.top))

def multidim_search(rec: Rec, is_member) -> [({Rec}, {Rec}), ]:
    """Generator for iteratively approximating the oracle's threshold."""
    initial_vol = unknown_vol = volume(rec)
    queue = [(unknown_vol, rec)]
    good_approx, bad_approx = [], []
    while queue:
        _, rec = hpop(queue)
        rec = Rec(*map(np.array, rec))
        low, mid, high = weightedbinsearch(rec, is_member)
        backward, forward, incomparables = subdivide(low, mid, high, rec)
        bad_approx.append(backward)
        good_approx.append(forward)
        
        for r in incomparables:
            hpush(queue, (-volume(r), to_tuple(r)))

        # not correct, since is doesn't include upward closure's area
        unknown_vol -= volume(backward) + volume(forward)
        est_pct_vol = unknown_vol / initial_vol
        yield est_pct_vol, (bad_approx, good_approx, queue)


def draw_rec(dwg, r: Rec, is_member: bool):
    """TODO: handle different orientations."""
    bot = tuple(map(float, r.bot))
    dim = tuple(map(float, r.top - r.bot))
    color = "red" if is_member else "green"
    return dwg.rect(bot, dim, fill=color)


def draw_domain(r: Rec, good: {Rec}, bad: {Rec}, scale):
    width, height = tuple(map(float, scale(r.top - r.bot)))
    dwg = svgwrite.Drawing(width=width, height=height)
    scale_rec = lambda x: Rec(scale(x.bot), scale(x.top))
    good_recs = (draw_rec(dwg, scale_rec(r), True) for r in good)
    bad_recs = (draw_rec(dwg, scale_rec(r), False) for r in bad)

    for svg_rec in chain(good_recs, bad_recs):
        dwg.add(svg_rec)
    return dwg



def multidim_search_and_draw(rec, is_member, save_path=None,
                             *, vol_tol=0.02, n=1000):
    def f(x):
        return x[0] > n or float(x[1][0]) < vol_tol
    approxes = multidim_search(rec, is_member)
    _, (_, (bad, good, q)) = fn.first(filter(f, enumerate(approxes)))

    # TODO automate detecting a good scale
    # currently assumes -1, 1 to 0, 100 transformation
    scale = lambda x: 100 * (x + 1)
    dwg = draw_domain(rec, good=good, bad=bad, scale=scale)
    if save_path:
        with Path(save_path).open('w') as f:
            dwg.write(f)
    return dwg


def main():
    R = Rec(-np.ones(2), np.ones(2))
    n = np.array([1, 1]) / np.sqrt(2)
    f = lambda x: x @n > 0
    multidim_search_and_draw(R, f, "foo.svg")

    f = lambda x: x[0] > 0
    multidim_search_and_draw(R, f, "foo2.svg")

    f = lambda x: x[1] > 0
    multidim_search_and_draw(R, f, "foo3.svg")

    f = lambda x: (x @n > 0 and x[0] > 0) or (x @n > 0.2 and x[0] < 0)
    multidim_search_and_draw(R, f, "foo4.svg")

    f = lambda x: np.abs(x[1]) > x[0]**2 if x[0] < 0 else x@n > 0
    multidim_search_and_draw(R, f, "foo5.svg")

    

if __name__ == "__main__":
    main()
