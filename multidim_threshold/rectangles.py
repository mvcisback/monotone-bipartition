from typing import NamedTuple, Tuple, Iterable, Hashable

import funcy as fn
from lenses import lens

class Interval(NamedTuple):
    bot: float
    top: float


class Rec(NamedTuple):
    intervals: Iterable[Interval]
    tag: Hashable
    error: float
    
    @property
    def bot(self):
        return tuple(fn.pluck(0, self.intervals))

    @property
    def top(self):
        return tuple(fn.pluck(1, self.intervals))

    @property
    def diag(self):
        return tuple(t-b for b, t in zip(self.bot, self.top))
Rec.__new__.__defaults__ = (None, None)

intervals_lens = lens.GetAttr('intervals')
tag_lens = lens.GetAttr('tag')
