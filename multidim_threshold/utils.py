from collections import namedtuple, Iterable

import funcy as fn
import numpy as np
from lenses import lens

Result = namedtuple("Result", "unexplored")
Rec = namedtuple("Rec", "bot top")

map_array = fn.partial(map, np.array)
map_tuple = fn.partial(map, tuple)


def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def volume(rec: Rec):
    return np.prod(np.abs(rec.bot - rec.top))


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


def naive_hausdorff(res1, res2):
    X, Y = res1.mids, res2.mids
    return max(_d(X, Y), _d(Y, X)) 

def _d(X, Y):
    return max(d(x, Y) for x in X)

def d(x, Y):
    return min(np.linalg.norm(np.array(x) - np.array(y), np.inf) for y in Y)
