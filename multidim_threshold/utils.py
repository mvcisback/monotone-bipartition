from itertools import chain, product
from collections import namedtuple, Iterable, defaultdict

import funcy as fn
import numpy as np

from multidim_threshold.rectangles import Rec

Result = namedtuple("Result", "unexplored")

map_tuple = fn.partial(map, tuple)


def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def volume(rec: Rec):
    return np.prod(np.array(rec.top) - np.array(rec.bot))


def infinity_volume(rec: Rec):
    return max(rec.diag)


def smallest_edge(rec: Rec):
    return min(rec.diag)


def longest_edge(rec: Rec):
    return max(rec.diag)


def avg_edge(rec: Rec):
    return sum(rec.diag) / len(rec.diag)


def basis_vec(i, dim):
    """Basis vector i"""
    a = np.zeros(dim)
    a[i] = 1.0
    return a


@fn.memoize
def basis_vecs(dim):
    """Standard orthonormal basis."""
    return [basis_vec(i, dim) for i in range(dim)]


def degenerate(r):
    return any(x == 0 for x in r.diag)


def contains(r1, r2):
    return all(a1 >= a2 and b1 <= b2
               for a1, a2, b1, b2 in zip(r1.top, r2.top, r1.bot, r2.bot))
