from nose2.tools import params
import unittest

import multidim_threshold as mdt
import numpy as np

r0 = mdt.Rec(np.array([-1]), np.array([1]))
p0 = np.array([0])
f0 = mdt.Rec(np.array([0]), np.array([1]))
b0 = mdt.Rec(np.array([-1]), np.array([0]))
i0 = set()
n0 = np.array([1])
F0 = lambda x: x@n0 > 0

r1 = mdt.Rec(np.array([-1,-2]), np.array([1,2]))
p1 = np.array([0, 0])
f1 = mdt.Rec(np.array([0, 0]), np.array([1, 2]))
b1 = mdt.Rec(np.array([-1, -2]), np.array([0, 0]))
i1 = {mdt.Rec((0, -2), (1, 0)), mdt.Rec((-1, 0), (0, 2))}
n1 = np.array([1, 1]) / np.sqrt(2)
F1 = lambda x: x @n1 > 0

r2 = mdt.Rec(np.array([-2,-1, -2]), np.array([3,1, 3]))
p2 = np.array([0, 0, 0])
f2 = mdt.Rec(np.array([0, 0, 0]), np.array([3, 1, 3]))
b2 = mdt.Rec(np.array([-2, -1, -2]), np.array([0, 0, 0]))
n2 = np.array([1, 1, 1]) / np.sqrt(3)
F2 = lambda x: x @n2 > 0


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
