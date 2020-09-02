import operator as op
from itertools import product
from functools import reduce
from typing import Iterable, NamedTuple

import attr
import numpy as np
from lenses import lens


bot_lens = lens.intervals.Each().bot
top_lens = lens.intervals.Each().top
intervals_lens = lens.GetAttr('intervals')


@attr.s(frozen=True, auto_attribs=True)
class Interval:
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

    @property
    def center(self):
        return self.bot + self.radius/2

    def __iter__(self):
        yield from (self.bot, self.top)


def _select_rec(intervals, j, lo, hi):
    def include_error(i, k, l, h):  # noqa: E741
        idx = (j >> k) & 1
        l2, h2 = i[idx]
        return min(l2, l), max(h, h2)

    chosen_rec = tuple(
        include_error(i, k, l, h)
        for k, (l, h, i) in enumerate(zip(lo, hi, intervals)))
    error = max(h - l for h, l in zip(hi, lo))
    return to_rec(chosen_rec, error=error)


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
        return max(x for x in r.diag) == 0

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
        return to_rec(
            [i1 & i2 for i1, i2 in zip(self.intervals, other.intervals)]
        )

    def sup(self, other):
        return to_rec(
            [i1 | i2 for i1, i2 in zip(self.intervals, other.intervals)]
        )

    def discretize(self, eps=3):
        return list(product(*(i.discretize(eps) for i in self.intervals)))

    @property
    def shortest_edge(self):
        return min(self.diag)

    @property
    def corners(self):
        return frozenset(product(*self.intervals))

    @property
    def center(self):
        return tuple(i.center for i in self.intervals)


def to_rec(intervals, error=0):
    intervals = tuple(Interval(*i) for i in intervals)
    return Rec(intervals, error)


def unit_rec(n):
    return to_rec([[0, 1]]*n)
