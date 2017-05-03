"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import combinations
from heapq import heappush as hpush, heappop as hpop

import numpy as np
from numpy import array
import funcy as fn
import networkx as nx
from intervaltree import IntervalTree, Interval

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


def generate_incomparables(mid, r):
    """Generate all incomparable hyper-boxes."""
    forward, backward = forward_cone(mid, r), backward_cone(mid, r)
    bases = (backward.bot, forward.bot)
    diags = (backward.top - backward.bot, forward.top - forward.bot)
    dim = len(bases[0])
    basis = basis_vecs(dim)
    for i in range(1, dim):
        for es in combinations(basis, i):
            vs = tuple(sum((diag @e) * e for e in es) for diag in diags)
            yield Rec(*[base + v for base, v in zip(bases, vs)])


def subdivide(low, mid, high, r: Rec) -> [Rec]:
    """Computes the set of incomparable cones of point p."""
    incomparables = list(generate_incomparables(mid, r))
    return backward_cone(low, r), forward_cone(high, r), incomparables


def refine(rec: Rec, diagsearch):
    low, mid, high = diagsearch(rec)
    if mid is None:
        mid = low
    _, _, incomparables = subdivide(low, mid, high, rec)
    return incomparables
    

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
    # Co-routines for refining rectangles
    refiners = [_refiner(lo, hi, oracle) for oracle in oracles]

    # First approximations
    rec_sets = [next(refiner) for refiner in refiners]

    # Initial set of edges
    edges = set(map(frozenset, combinations(range(len(oracles)), 2)))

    g = nx.Graph()
    # Create Adacjency Graph
    for edge in edges:
        i, j = edge
        recs_i, recs_j = rec_sets[i], rec_sets[j]
        # TODO: implement more sophisticated blame tracking
        pH, _ = mdt.rectangleset_pH(recs_i, recs_j)
        dH, _ = mdt.rectangleset_dH(recs_i, recs_j)
        if pH == dH:
            # TODO: hack. IntervalTree doesn't support 0 points
            # So we add an interval with smaller than tolerance
            # length
            dH += tol/3
        g.add_edge(i, j, interval=Interval(pH, dH, edge))

    # Create Interval Tree
    t = IntervalTree(fn.pluck(2, g.edges_iter(data="interval")))

    while len(g) != 1:
        yield g, t
        can_merge, result = mdt.clusters_to_merge(t, tol)
        if can_merge:
            i, j = result
            mdt.merge_clusters(v1=i, v2=j, tree=t, graph=g)
        else:
            raise NotImplementedError
    yield g, t

