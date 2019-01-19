from functools import partial, total_ordering

import attr
import funcy as fn
from lazytree import LazyTree

from monotone_bipartition import hausdorff as mbph
from monotone_bipartition import rectangles  # unit_rec
from monotone_bipartition import refine as mbpr  # bounding_box
from monotone_bipartition import search as mdts  # binsearch


@attr.s
@total_ordering
class BiPartition:
    tree: LazyTree = attr.ib()
    func = attr.ib()

    @property
    def dim(self):
        return len(self.tree.view())

    def approx(self, tol=1e-4):
        recs = self.tree \
                   .prune(isleaf=lambda r: r.shortest_edge <= tol) \
                   .leaves()

        return list(recs)

    def dist(self, other, tol=1e-4) -> rectangles.Interval:
        approxes = mbph.gen_dists(self, other)
        within_tol = (i for i in approxes if i.radius < tol)
        return fn.first(within_tol)

    def label(self, point) -> bool:
        # TODO: Should support either finite precision or max depth.
        domain = rectangles.unit_rec(self.dim)

        def above(rec):
            return point in domain.forward_cone(rec.bot) and \
                point not in domain.backward_cone(rec.top)

        def below(rec):
            return point not in domain.forward_cone(rec.bot) and \
                point in domain.backward_cone(rec.top)

        def not_inside(rec):
            return point not in domain.forward_cone(rec.bot) or \
                point not in domain.backward_cone(rec.top)

        recs = self.tree.prune(isleaf=not_inside).bfs()
        for rec in recs:
            if above(rec):
                return True

            if not not_inside(rec):
                if all(x == 0 for x in rec.diag):  # point rec.
                    return True
            elif below(rec):
                return all(x == 0 for x in rec.bot)

        raise RuntimeError("Point outside domain?!?!?!")

    def __le__(self, other):
        raise NotImplementedError

    def project(self, point_or_ordering, *, lexicographic=False, tol=1e-4):
        """
        If lexicographic is False, then returns an approximation
        to the *unique* point that intersect the threshold
        boundary AND the line intersecting the origin and the
        user defined point.

        If lexicographic is True, then returns an approximation to the
        minimum point on the threshold boundary that is minimum in the
        ordering provided. The ordering is given as a list of pairs:
        `(index, minimize)`. The index is minimized if `minimize` is
        true, and maximized otherwise. For example, `[(1, False), (0,
        True)]` encodes maximizing along axis 1 of the unit box and
        then minimizing along axis 0.
        """
        if lexicographic:
            return mdts.lexicographic_opt(self.func, point_or_ordering, tol)
        else:
            return mdts.line_intersect(self.func, point_or_ordering, tol)


def from_threshold(func, dim: int, *, memoize_nodes=True) -> BiPartition:
    bounding_box = mbpr.bounding_box(rectangles.unit_rec(dim), func)
    diagsearch = partial(mdts.binsearch, oracle=func)
    refine = partial(mbpr.refine, diagsearch=diagsearch)
    if memoize_nodes:
        refine = fn.memoize()(refine)

    return BiPartition(LazyTree(root=bounding_box, child_map=refine), func)
