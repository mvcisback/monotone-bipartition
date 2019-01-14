from functools import partial

import attr
from lazytree import LazyTree

from monotone_bipartition import rectangles  # unit_rec
from monotone_bipartition import refine as mbpr  # bounding_box
from monotone_bipartition import search as mdts  # binsearch


@attr.s
class BiPartition:
    tree: LazyTree = attr.ib()

    @property
    def dim(self):
        return len(self.tree.view())

    @property
    def domain(self):
        return self.tree.view()
    
    def dist(self, other) -> float:
        raise NotImplementedError

    def label(self, point) -> bool:
        # TODO: Should support either finite precision or max depth.
        domain = self.domain

        def not_comparable(rec, point):
            return point not in domain.forward_cone(rec.bot) and \
                point not in domain.backward_cone(rec.top)

        recs = self.tree.prune(not_comparable).leaves()
        for rec in recs:
            if rec in domain.forward_cone(rec.top):
                return True
            elif rec in domain.backward_cone(rec.bot):
                return False


def from_threshold(func, dim: int) -> BiPartition:
    bounding_box = mbpr.bounding_box(rectangles.unit_rec(dim), func)
    diagsearch = partial(mdts.binsearch, oracle=func)
    refine = partial(mbpr.refine, diagsearch=diagsearch)
    return BiPartition(LazyTree(
        root=bounding_box,
        child_map=refine,
    ))
