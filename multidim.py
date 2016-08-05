"""Implements muli-dimensional threshold discovery via binary search."""
from collections import namedtuple, deque
from itertools import chain
from pathlib import Path

from numpy import array
import svgwrite
import numpy as np
import funcy as fn

Rec = namedtuple("Rec", "bot top")


def binsearch(r: Rec, is_member, eps=0.001) -> (array, array, array):
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


def forward_cone(p: array, r: Rec) -> Rec:
    """Computes the forward cone from point p."""
    return Rec(p, r.top)


def backward_cone(p: array, r: Rec) -> Rec:
    """Computes the backward cone from point p."""
    return Rec(r.bot, p)


def incomparable(p: array, r: Rec) -> [Rec]:
    """Computes the set of incomparable cones of point p."""
    r01 = Rec(array([r.bot[0], p[1]]), array([p[0], r.top[1]]))
    r10 = Rec(array([p[0], r.bot[1]]), array([r.top[0], p[1]]))
    return [r01, r10]


def multidim_search(rec: Rec, is_member) -> [({Rec}, {Rec})]:
    """Generator for iteratively approximating the oracle's threshold."""
    queue = deque([rec])
    good_approx, bad_approx = [], []
    while True:
        rec = queue.pop()
        low, mid, high = binsearch(rec, is_member)

        bad_approx.append(backward_cone(low, rec))
        good_approx.append(forward_cone(mid, rec))
        queue.extendleft(incomparable(mid, rec))

        yield bad_approx, good_approx


def draw_rec(dwg, r:Rec, is_member:bool):
    """TODO: handle different orientations."""
    bot = tuple(map(float, r.bot))
    dim = tuple(map(float, r.top - r.bot))
    color = "red" if is_member else "green"
    return dwg.rect(bot, dim, fill=color)


def draw_domain(r:Rec, good:{Rec}, bad:{Rec}, scale):
    width, height = tuple(map(float, scale(r.top - r.bot)))
    dwg = svgwrite.Drawing(width=width, height=height)
    scale_rec = lambda x: Rec(scale(x.bot), scale(x.top))
    good_recs = (draw_rec(dwg, scale_rec(r), True) for r in good)
    bad_recs = (draw_rec(dwg, scale_rec(r), False) for r in bad)

    for svg_rec in chain(good_recs, bad_recs):
        dwg.add(svg_rec)
    return dwg


def multidim_search_and_draw(rec, is_member, n, save_path):
    bad, good = fn.nth(n, multidim_search(rec, is_member))
    # TODO automate detecting a good scale
    # currently assumes -1, 1 to 0, 100 transformation
    scale = lambda x: 100*(x +1)
    dwg = draw_domain(rec, good=good, bad=bad, scale=scale)
    with Path(save_path).open('w') as f:
        dwg.write(f)


def main():
    R = Rec(-np.ones(2), np.ones(2))
    n = np.array([1,1])/np.sqrt(2)
    f = lambda x: x@n > 0
    multidim_search_and_draw(R, f, 100, "foo.svg")

    f = lambda x: x[0] > 0
    multidim_search_and_draw(R, f, 100, "foo2.svg")

    f = lambda x: x[1] > 0
    multidim_search_and_draw(R, f, 100, "foo3.svg")

    f = lambda x: (x@n > 0 and x[0] > 0) or (x@n > -0.2 and x[0] < 0)
    multidim_search_and_draw(R, f, 100, "foo4.svg")

        

if __name__ == "__main__":
    main()
    
