from typing import NamedTuple, Tuple, Iterable, Hashable

import funcy as fn

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
Rec.__new__.__defaults__ = (None, None)



