from typing import NamedTuple, Tuple, Iterable

import funcy as fn

class Interval(NamedTuple):
    bot: float
    top: float

class Rec(NamedTuple):
    intervals: Iterable[Interval]
    
    @property
    def bot(self):
        return tuple(fn.pluck(0, self.intervals))

    @property
    def top(self):
        return tuple(fn.pluck(1, self.intervals))
