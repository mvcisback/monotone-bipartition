import operator as op
from itertools import product
from functools import reduce
from typing import Iterable, NamedTuple
from enum import Enum

import funcy as fn
import numpy as np
from lenses import lens

from monotone_bipartition import search as mdts  # SearchResultType, binsearch
from monotone_bipartition import refine as mdtr


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

    def __and__(self, other):
        return to_rec([i1 & i2 for i1, i2 in zip(self.intervals, other.intervals)])

    def sup(self, other):
        return to_rec([i1 | i2 for i1, i2 in zip(self.intervals, other.intervals)])

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
