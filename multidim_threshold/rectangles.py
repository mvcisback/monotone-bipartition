import operator as op
from itertools import product
from functools import reduce
from typing import Iterable, NamedTuple
from enum import Enum

import funcy as fn
import numpy as np
from lenses import lens

from multidim_threshold import search as mdts  # SearchResultType, binsearch
from multidim_threshold import refine as mdtr


bot_lens = lens.intervals.Each().bot
top_lens = lens.intervals.Each().top
intervals_lens = lens.GetAttr('intervals')


class CMP(Enum):
    ForwardCone = 1
    BackwardCone = 2
    Inside = 3
    Incomparable = 4


class Interval(NamedTuple):
    bot: float
    top: float

    def __contains__(self, x):
        if isinstance(x, Interval):
            return self.bot <= x.bot and x.top <= self.top
        return self.bot <= x <= self.top

    def __and__(self, i2):
        bot, top = max(i2.bot, self.bot), min(i2.top, self.top)
        if bot > top:
            return None
        return Interval(bot, top)

    def __or__(self, i2):
        bot, top = min(i2.bot, self.bot), max(i2.top, self.top)
        return Interval(bot, top)

    def discretize(self, k=3):
        return np.linspace(self.bot, self.top, k)

    @property
    def radius(self):
        return self.top - self.bot

    def label(self, point):
        if point < self.bot:
            return CMP.BackwardCone
        elif point > self.top:
            return CMP.ForwardCone
        elif self.bot <= point <= self.top:
            return CMP.Inside
        else:
            return CMP.Incomparable


def _select_rec(intervals, j, lo, hi):
    def include_error(i, k, l, h):
        idx = (j >> k) & 1
        l2, h2 = i[idx]
        return min(l2, l), max(h, h2)

    chosen_rec = tuple(
        include_error(i, k, l, h)
        for k, (l, h, i) in enumerate(zip(lo, hi, intervals)))
    error = max(h - l for h, l in zip(hi, lo))
    return to_rec(chosen_rec, error=error)


def _join_itvl_labels(l1, l2):
    if l1 == l2:
        return l1
    elif CMP.Incomparable in (l1, l2):
        return CMP.Incomparable
    elif l1 == CMP.Inside:
        return l2
    elif l2 == CMP.Inside:
        return l1
    else:
        return CMP.Incomparable


class Rec(NamedTuple):
    intervals: Iterable[Interval]
    error: float

    @property
    def bot(self):
        return bot_lens.collect()(self)

    @property
    def top(self):
        return top_lens.collect()(self)

    @property
    def diag(self):
        return tuple(t - b for b, t in zip(self.bot, self.top))

    @property
    def dim(self):
        return len(self.intervals)

    @property
    def volume(self):
        return reduce(op.mul, lens.intervals.Each().radius.collect()(self))

    @property
    def degenerate(r):
        return min(x for x in r.diag) < 1e-3

    @property
    def is_point(r):
        return max(x for x in r.diag) < 1e-3

    def forward_cone(self, p):
        """Computes the forward cone from point p."""
        return to_rec(zip(p, self.top))

    def backward_cone(self, p):
        """Computes the backward cone from point p."""
        return to_rec(zip(self.bot, p))

    def subdivide(self, rec2, drop_fb=True):
        """Generate all 2^n - 2 incomparable hyper-boxes.
        TODO: Do not generate unnecessary dimensions for degenerate surfaces
        """
        n = self.dim
        if n <= 1:
            return
        elif drop_fb:
            indicies = range(1, 2**n - 1)
        else:
            indicies = range(0, 2**n)
        lo, hi = rec2.bot, rec2.top
        forward, backward = self.forward_cone(lo), self.backward_cone(hi)
        intervals = list(zip(backward.intervals, forward.intervals))
        x = {_select_rec(intervals, j, lo, hi) for j in indicies}
        yield from x - {self}

    def __contains__(self, r):
        if not isinstance(r, Rec):
            return all(p in i for i, p in zip(self.intervals, r))
        return all(i2 in i1 for i1, i2 in zip(self.intervals, r.intervals))

    def discretize(self, eps=3):
        return list(product(*(i.discretize(eps) for i in self.intervals)))

    @property
    def shortest_edge(self):
        return min(self.diag)

    def label(self, point):
        pt_intervals = zip(point, self.intervals)
        return reduce(_join_itvl_labels, (i.label(p) for p, i in pt_intervals))


def to_rec(intervals, error=0):
    intervals = tuple(Interval(*i) for i in intervals)
    return Rec(intervals, error)


def unit_rec(n):
    return to_rec([[0, 1]]*n)


class _RecTree:
    def __init__(self, data, oracle):
        self.data = data
        self.oracle = oracle

    @property
    @fn.memoize
    def children(self):
        children = (r for r in mdtr.refine(self.data, self.oracle))
        return fn.lmap(lambda r: _RecTree(r, self.oracle), children)

    def __lt__(self, other):
        return self.data < other.data

    def label(self, point, approx=True, max_depth=10):
        if max_depth <= 0:
            bot, top = map(np.array, (self.data.bot, self.data.top))
            if (np.array(point) > (top + bot)/2).all():
                return CMP.ForwardCone
            else:
                return CMP.BackwardCone

        # See if any of the children can for sure label the point.
        for child in self.children:
            lbl = child.data.label(point)
            if lbl in (CMP.ForwardCone, CMP.BackwardCone):
                return lbl
            elif lbl == CMP.Inside:
                return child.label(point, approx, max_depth-1)

        # BFS for label.
        for child in self.children:
            lbl = child.label(point, approx, max_depth-1)
            if lbl != CMP.Incomparable:
                return lbl


class RecTree(_RecTree):
    def __init__(self, n, oracle):
        data = mdtr.bounding_box(unit_rec(n), oracle)
        oracle = fn.partial(mdts.binsearch, oracle=oracle)
        super().__init__(data=data, oracle=oracle)

    def dist(self, other, eps=1e-4, avg=True):
        d_bounds = mdtr.hausdorff_bounds(self, other, eps=eps)
        d_itvl = fn.first(itvl for itvl in d_bounds if itvl.radius < eps)
        return sum(d_itvl) / d_itvl.radius if avg else d_itvl 
