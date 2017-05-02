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
from multidim_threshold.search import binsearch, weightedbinsearch
from multidim_threshold.projection import find_boundaries


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
    

def _refiner(lo, hi, oracle, diagsearch=None):
    """Generator for iteratively approximating the oracle's threshold."""
    rec = to_rec(lo, hi)

    if diagsearch is None:
        bool_oracle = isinstance(oracle(rec.bot), bool)
        diagsearch = binsearch if bool_oracle else weightedbinsearch
        diagsearch = fn.partial(diagsearch, oracle=oracle)

    rec = yield [find_boundaries(rec, diagsearch)]
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


def hausdorff_guided_clustering(los, his, oracles, diagsearches, tol=1e-3):
    # Initial set of edges
    edges = set(map(frozenset, combinations(nodes, 2)))

    # Co-routines for refining rectangles
    refiners = [_refiner(*args) for args in zip(los, his, oracles, diagsearches)]

    # First approximations
    rec_sets = [next(refiner) for refiner in refiners]

    # Map from rectangles to which edges is watching the rectangle
    watching = {(rec_sets[0], i): edges  for i, rec_set in enumerate(rec_sets)}

    g = nx.Graph()

    # Create Adacjency Graph
    for edge in edges:
        i, j = edge
        recs_i, recs_j = recs_sets[i], recs_sets[j]
        pH = mdt.rectangleset_pH(recs_i, recs_j)
        dH = mdt.rectangleset_dH(recs_i, recs_j)
        g.add_edge(i, j, interval=Interval(ph, dH, edge))

    # Create Interval Tree
    t = IntervalTree(fn.pluck(2, g.edges_iter(data="interval")))
    # Until Tree is created keep refining
    while len(g) != 1:
        yield g, t
        can_merge, result = mdt.clusters_to_merge(t)
        if can_merge:
            i, j = result
            mdt.merge_clusters(v1=i, v2=j, tree=t, graph=g)
        else:
            (i, j), eps = result
            l = l2 = float('inf') # TODO: set to current length of smallest interval
            
            while l - l2 < max(eps, tol):
                # TODO: lookup which rectangles are causing the hausdorff distance!
                "refine relevant rectangle set based on rectangle causing hasudorff"
