from itertools import chain, product
from collections import namedtuple, Iterable

import funcy as fn
import numpy as np

from intervaltree import IntervalTree, Interval

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


def diff_dimensions(vec1, vec2):
    return (abs(a - b) for a, b in zip(vec1, vec2))


def diff_tops(r1: Rec, r2: Rec):
    return diff_dimensions(r1.top, r2.top)


def diff_bots(r1: Rec, r2: Rec):
    return diff_dimensions(r1.bot, r2.bot)


def rectangle_hausdorff(r1: Rec, r2: Rec):
    return max(chain(diff_bots(r1, r2), diff_tops(r1, r2)))


def rectangle_pH(r1: Rec, r2: Rec):
    return max(map(min, zip(diff_bots(r1, r2), diff_tops(r1, r2))))


def _argminmax(A, axis):
    idxs = A.argmin(axis=axis)
    idx2 = np.choose(idxs, A.T).argmax()
    idx1 = idxs[idx2]
    if axis == 1:
        idx1, idx2 = idx2, idx1
    return A[idx1, idx2], (idxs[idx2], idx2)


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


def clusters_to_merge(tree, tol=1e-4):
    # Select which two clusters to merge
    min_intervals = tree[tree.begin()]

    first = min(min_intervals)
    if len(min_intervals) == 1 or first.length() < tol:
        return True, first.data

    refinement_len = max(overlap_len(first, i) for i in tree[first] if i != first)
    return False, (first.data, refinement_len)


def merge_clusters(v1, v2, tree, graph):
    v12 = frozenset([v1, v2])
    graph.add_node(v12)
    tree.remove(graph[v1][v2]["interval"])
    for v3 in (v for v in graph.neighbors(v1) if v != v2):
        i1, i2 = graph[v1][v3]["interval"], graph[v2][v3]["interval"]
        
        # Remove Intervals from interval tree
        tree.remove(i1)
        tree.remove(i2)

        # Compute merged interval and add to tree and graph
        merged_interval = Interval(
            max(i1.begin, i2.begin), max(i1.end, i2.end), {v12, v3})
        tree.add(merged_interval)
        graph.add_edge(v12, v3, interval=merged_interval)

    # Remove merged clusters
    graph.remove_node(v1)
    graph.remove_node(v2)
