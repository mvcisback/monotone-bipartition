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
    return np.prod(np.array(rec.top) - np.array(rec.bot))


def basis_vec(i, dim):
    """Basis vector i"""
    a = np.zeros(dim)
    a[i] = 1.0
    return a


@fn.memoize
def basis_vecs(dim):
    """Standard orthonormal basis."""
    return [basis_vec(i, dim) for i in range(dim)]


def bounding_rec(recs):
    recs = np.array(list(recs))
    return Rec(recs.min(axis=0), recs.max(axis=0))
