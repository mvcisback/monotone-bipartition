"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import combinations
from heapq import heappush as hpush, heappop as hpop, heapify

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
    if (low == high).all():
        return [Rec(low, low)]
    return list(generate_incomparables(low, high, rec))
    

def bounding_box(r: Rec, oracle):
    diag = np.array(r.top) - np.array(r.bot)
    basis = basis_vecs(len(r.bot))
    recs = [Rec(r.bot, r.bot + diag*v) for v in basis]
    top = np.array([binsearch(r2, oracle)[2]@v for v, r2 in zip(basis, recs)])
    return Rec(bot=r.bot, top=top)


def _refiner(oracle, diagsearch=None):
    """Generator for iteratively approximating the oracle's threshold."""
    diagsearch = fn.partial(binsearch, oracle=oracle)    
    rec = yield
    while True:
        rec = yield refine(rec, diagsearch)


def guided_refinement(rec_set, oracles, cost, prune=lambda *_: False, 
                      diagsearch=None):
    """Generator for iteratively approximating the oracle's threshold."""
    refiners = {k: _refiner(o) for k, o in oracles.items()}
    queue = [(cost(rec, tag), (tag, to_tuple(rec))) for tag, rec in rec_set 
             if not prune(rec, tag)]
    heapify(queue)
    for refiner in refiners.values():
        next(refiner)

    while queue:
        yield queue
        _, (tag, rec) = hpop(queue)
        for r in refiners[tag].send(Rec(*map(np.array, rec))):
            if prune(r, tag):
                continue
            hpush(queue, (cost(r, tag), (tag, to_tuple(r))))


def volume_guided_refinement(rec_set, oracles, diagsearch=None):
    return guided_refinement(rec_set, oracles, lambda r, _: volume(r), diagsearch=diagsearch)
