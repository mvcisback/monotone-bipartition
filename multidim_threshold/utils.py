from itertools import chain, product
from collections import namedtuple, Iterable

import funcy as fn
import numpy as np

Result = namedtuple("Result", "unexplored")
Rec = namedtuple("Rec", "bot top")

map_tuple = fn.partial(map, tuple)


def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def volume(rec: Rec):
    return np.prod(np.abs(rec.bot - rec.top))


def bounding_rec(recs):
    recs = np.array(list(recs))
    return Rec(recs.min(axis=0), recs.max(axis=0))


def _lift_rectangle_distance(recs1, recs2, rect_dist):
    """Naive implementation
    TODO: update with smarter exploiting updates and partialorder."""
    incident = np.zeros((len(recs1), len(recs2)))
    for (i, x), (j, y) in product(enumerate(recs1), enumerate(recs2)):
        incident[i, j] = rect_dist(x, y)

    return max(_argminmax(incident, 0), _argminmax(incident, 1))


def rectangleset_dH(recs1, recs2):
    return _lift_rectangle_distance(recs1, recs2, rectangle_hausdorff)


def rectangleset_pH(recs1, recs2):
    return _lift_rectangle_distance(recs1, recs2, rectangle_pH)


def overlap_len(i1, i2):
    return max(0, min(i1.end, i2.end) - max(i1.begin, i1.begin))


