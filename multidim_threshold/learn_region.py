"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import combinations
from heapq import heappush as hpush, heappop as hpop

import numpy as np
from numpy import array
import funcy as fn

import multidim_threshold as mdt
from multidim_threshold.utils import Result, Rec, to_rec, volume, basis_vecs
from multidim_threshold.search import binsearch


def to_tuple(r: Rec):
    return Rec(*map(tuple, r))


def forward_cone(p: array, r: Rec) -> Rec:
    """Computes the forward cone from point p."""
    return Rec(p, r.top)


def backward_cone(p: array, r: Rec) -> Rec:
    """Computes the backward cone from point p."""
    return Rec(r.bot, p)


def generate_incomparables(lo, hi, r):
    """Generate all incomparable hyper-boxes."""
    forward, backward = forward_cone(lo, r), backward_cone(hi, r)
    bases = (backward.bot, forward.bot)
    diags = (backward.top - backward.bot, forward.top - forward.bot)
    dim = len(bases[0])
    basis = basis_vecs(dim)
    for i in range(1, dim):
        for es in combinations(basis, i):
            vs = tuple(sum((diag @e) * e for e in es) for diag in diags)
            yield Rec(*[base + v for base, v in zip(bases, vs)])


def refine(rec: Rec, diagsearch):
    low, _, high = diagsearch(rec)
    return list(generate_incomparables(low, high, rec))
    

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


def cost_guided_refinement(lo, hi, oracle, cost, diagsearch=None):
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


def volume_guided_refinement(lo, hi, oracle, diagsearch=None):
    yield from cost_guided_refinement(lo, hi, oracle, volume, diagsearch)
