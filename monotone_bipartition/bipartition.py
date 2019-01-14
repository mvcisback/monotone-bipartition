from functools import partial

import attr
from lazytree import LazyTree

from monotone_bipartition import rectangles  # unit_rec
from monotone_bipartition import refine as mbpr  # bounding_box
from monotone_bipartition import search as mdts  # binsearch


@attr.s
class BiPartition:
    tree: LazyTree = attr.ib()

    def dist(self, other) -> float:
        raise NotImplementedError

    def label(self, point) -> bool:
        raise NotImplementedError


def from_threshold(func, dim: int) -> BiPartition:
    bounding_box = mbpr.bounding_box(rectangles.unit_rec(dim), func)
    diagsearch = fn.partial(mdts.binsearch, oracle=func)
    refine = lambda r: mbpr.refine(r, diagsearch)
    return BiPartition(LazyTree(
        root=bounding_box,
        child_map=refine,
    ))
