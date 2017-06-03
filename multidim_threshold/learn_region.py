"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import product
from heapq import heappush as hpush, heappop as hpop

import numpy as np
from numpy import array
import funcy as fn

import multidim_threshold as mdt
from multidim_threshold.utils import Result, Rec, to_rec, volume
from multidim_threshold.search import binsearch


def to_tuple(r: Rec):
    return Rec(*map(tuple, r))


def intvls_to_rec(intvls):
    return Rec(*zip(*intvls))


@fn.autocurry
def not_comparable(lo, hi, r):
    """Returns True iff is not comparable (product ordering) to lo or hi."""
    return (r.bot > hi).any() or (r.bot < lo).any()


def subdivide(lo, hi, r: Rec) -> [Rec]:
    """Subdivides r at (lo, hi) and returns the set of incomparable cones."""
    subdivide1d = [[(b, h), (l, t)] for b, l, h, t in zip(r.bot, lo, hi, r.top)]
    all_recs = map(intvls_to_rec, product(*subdivide1d))
    return list(filter(not_comparable(lo, hi), all_recs))


def refine(rec: Rec, diagsearch):
    low, _, high = diagsearch(rec)
    return subdivide(low, high, rec)
    

def bounding_box(r: Rec, oracle):
    diag = np.array(r.top) - np.array(r.bot)
    basis = basis_vecs(len(r.bot))
    recs = [Rec(r.bot, r.bot + diag*v) for v in basis]
    top = np.array([binsearch(r2, oracle)[2]@v for v, r2 in zip(basis, recs)])
    return Rec(bot=r.bot, top=top)


def _refiner(lo, hi, oracle, diagsearch=None):
    """Generator for iteratively approximating the oracle's threshold."""
    rec = to_rec(lo, hi)
    diagsearch = fn.partial(binsearch, oracle=oracle)

    rec = yield [bounding_box(rec, oracle)]
    while True:
        rec = yield refine(rec, diagsearch)


def volume_guided_refinement(lo, hi, oracle, diagsearch=None):
    """Generator for iteratively approximating the oracle's threshold."""
    refiner = _refiner(lo, hi, oracle, diagsearch)
    rec = next(refiner)[0]
    queue = [(-volume(rec), to_tuple(rec))]

    while queue:
        yield Result(list(fn.pluck(1, queue)))
        _, rec = hpop(queue)
        refined = refiner.send(Rec(*map(np.array, rec)))
        for r in refined:
            hpush(queue, (-volume(r), to_tuple(r)))


def hausdorff_guided_clustering(lo, hi, oracles, tol=1e-6):
    # Waiting on new agglomerative clustering code
    raise NotImplementedError

