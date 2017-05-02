from nose2.tools import params
import unittest

import multidim_threshold as mdt
import numpy as np

from intervaltree import IntervalTree, Interval
import networkx as nx
from functools import partial

r0 = mdt.Rec(np.array([-1]), np.array([1]))
p0 = np.array([0])
f0 = mdt.Rec(np.array([0]), np.array([1]))
b0 = mdt.Rec(np.array([-1]), np.array([0]))
i0 = set()
n0 = np.array([1])
F0 = lambda x: x@n0 > 0

r1 = mdt.Rec(np.array([-1, -2]), np.array([1, 2]))
p1 = np.array([0, 0])
f1 = mdt.Rec(np.array([0, 0]), np.array([1, 2]))
b1 = mdt.Rec(np.array([-1, -2]), np.array([0, 0]))
i1 = {mdt.Rec((0, -2), (1, 0)), mdt.Rec((-1, 0), (0, 2))}
n1 = np.array([1, 1]) / np.sqrt(2)
F1 = lambda x: x @n1 > 0

r2 = mdt.Rec(np.array([-2, -1, -2]), np.array([3, 1, 3]))
p2 = np.array([0, 0, 0])
f2 = mdt.Rec(np.array([0, 0, 0]), np.array([3, 1, 3]))
b2 = mdt.Rec(np.array([-2, -1, -2]), np.array([0, 0, 0]))
n2 = np.array([1, 1, 1]) / np.sqrt(3)
F2 = lambda x: x @n2 > 0

ex1d = (lambda p: p > 5.435, lambda p: p - 5.435,
        mdt.Rec(np.array([-9]), np.array([9])))


class TestMultidimSearch(unittest.TestCase):

    @params((r0, 2), (r1, 8), (r2, 50))
    def test_volume(self, r, vol):
        self.assertEqual(mdt.volume(r), vol)

    @params((r0, p0, f0), (r1, p1, f1), (r2, p2, f2))
    def test_forward_cone(self, r, p, forward):
        forward = mdt.to_tuple(forward)
        forward2 = mdt.to_tuple(mdt.forward_cone(p, r))
        self.assertEqual(forward2, forward)

    @params((r0, p0, b0), (r1, p1, b1), (r2, p2, b2))
    def test_backward_cone(self, r, p, backward):
        backward = mdt.to_tuple(backward)
        backward2 = mdt.to_tuple(mdt.backward_cone(p, r))
        self.assertEqual(backward2, backward)

    @params((r0, p0, i0), (r1, p1, i1))
    def test_incomparables(self, r, mid, i):
        self.assertEqual(set(map(mdt.to_tuple,
                                 mdt.generate_incomparables(mid, r))), i)

    @params((r0, F0), (r1, F1))
    def test_binsearch(self, r, f):
        lo, mid, hi = mdt.binsearch(r, f, eps=0.0001)
        for i in mid:
            self.assertAlmostEqual(i, 0, places=3)

    @params(ex1d)
    def test_equiv_1d_mids(self, f, r, rec):
        lo1, _, hi1 = mdt.binsearch(rec, f, eps=0.001)
        lo2, _, hi2 = mdt.weightedbinsearch(rec, r, eps=0.01)
        self.assertAlmostEqual(float((lo1+hi1))/2, float((lo2 + hi1))/2, places=1)

    @params(ex1d)
    def test_polarity_invariant_1d(self, f, r, rec):
        neg_f = lambda p: not f(p)
        neg_r = lambda p: -r(p)

        _, mid1, _ = mdt.binsearch(rec, f, eps=0.0001)
        _, mid2, _ = mdt.binsearch(rec, neg_f, eps=0.0001)
        self.assertAlmostEqual(float(mid1 - mid2), 0, places=3)

        _, mid1, _ = mdt.weightedbinsearch(rec, r, eps=0.0001)
        _, mid2, _ = mdt.weightedbinsearch(rec, neg_r, eps=0.0001)
        self.assertAlmostEqual(float(mid1 - mid2), 0, places=3)


    def test_diff_dimensions(self):
        rec1 = mdt.Rec((0, 3), (2, 5))
        rec2 = mdt.Rec((3, 1), (5, 5))
        self.assertEqual(list(mdt.utils.diff_bots(rec1, rec2)), [3, 2])
        self.assertEqual(list(mdt.utils.diff_bots(rec2, rec1)), [3, 2])
        self.assertEqual(list(mdt.utils.diff_tops(rec2, rec1)), [3, 0])
        self.assertEqual(list(mdt.utils.diff_tops(rec1, rec2)), [3, 0])

    def test_rectangle_dH(self):
        rec1 = mdt.Rec((0, 3), (2, 5))
        rec2 = mdt.Rec((3, 1), (5, 5))

        self.assertEqual(mdt.utils.rectangle_hausdorff(rec1, rec2), 3)
        self.assertEqual(mdt.utils.rectangle_hausdorff(rec2, rec1), 3)

        
    def test_rectangle_pH(self):
        rec1 = mdt.Rec((0, 3), (2, 5))
        rec2 = mdt.Rec((3, 1), (5, 5))
        self.assertEqual(mdt.utils.rectangle_pH(rec1, rec2), 3)
        self.assertEqual(mdt.utils.rectangle_pH(rec2, rec1), 3)

        rec1 = mdt.Rec((0, 3), (2, 5))
        rec2 = mdt.Rec((3, 1), (2, 5))
        self.assertEqual(mdt.utils.rectangle_pH(rec1, rec2), 0)
        self.assertEqual(mdt.utils.rectangle_pH(rec2, rec1), 0)

    
    def test_rectangleset_dH(self):
        recs1 = [mdt.Rec((0, 3), (2, 5)), mdt.Rec((3, 1), (2, 5))]
        recs2 = [mdt.Rec((-2, 4), (5, 9)), mdt.Rec((2, 4), (12, 5))]
        self.assertEqual(mdt.utils.rectangleset_dH(recs1, recs2), 10)


    def test_rectangleset_pH(self):
        recs1 = [mdt.Rec((0, 3), (2, 5)), mdt.Rec((3, 1), (2, 5))]
        recs2 = [mdt.Rec((-2, 4), (5, 9)), mdt.Rec((2, 4), (12, 5))]
        self.assertEqual(mdt.utils.rectangleset_pH(recs1, recs2), 2)


    def test_clusters_to_merge(self):
        t1 = IntervalTree()
        t1[1:3] = 1
        t1[1:5] = 2
        t1[2:7] = 3
        t1[9:30] = 4
        can_merge, (iden, overlap_len) = mdt.utils.clusters_to_merge(t1)
        self.assertFalse(can_merge)
        self.assertEqual(overlap_len, 2)
        self.assertEqual(iden, 1)

        t2 = IntervalTree()
        t2[1:3] = 1
        t2[4:5] = 2
        t2[4:7] = 3
        t2[9:30] = 4
        can_merge, iden = mdt.utils.clusters_to_merge(t2)
        self.assertTrue(can_merge)
        self.assertEqual(iden, 1)


    def test_merging(self):
        g = nx.Graph()
        g.add_edge(1, 2, interval=Interval(1, 2, {1, 2}))
        g.add_edge(2, 3, interval=Interval(2, 6, {2, 3}))
        g.add_edge(3, 1, interval=Interval(3, 4, {3, 1}))

        t = IntervalTree()
        t[1:2] = {1, 2}
        t[2:6] = {2, 3}
        t[3:4] = {3, 1}

        mdt.utils.merge_clusters(1, 2, t, g)
        self.assertEqual({(1,2), 3}, set(g.nodes()))
        self.assertEqual(g[(1,2)][3]["interval"], Interval(3, 6, {(1,2), 3}))
        self.assertEqual(len(g.edges()), 1)
        self.assertEqual(len(t), 1)
        self.assertEqual(list(t[3:6])[0], Interval(3, 6, {(1,2), 3}))

        mdt.utils.merge_clusters((1,2), 3, t, g)
        self.assertEqual({((1,2), 3)}, set(g.nodes()))
        self.assertEqual(len(g.edges()), 0)
        self.assertEqual(len(t), 0)
