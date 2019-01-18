import funcy as fn

import monotone_bipartition as mbp
import monotone_bipartition.rectangles as mbpr
import monotone_bipartition.hausdorff as mdth


def test_rec_dist():
    r1 = mbpr.to_rec([(0, 1), (0.8, 0.8)])
    r2 = mbpr.to_rec([(0.6, 0.6), (0, 1)])
    assert 0.8 in mdth.rec_dist(r1, r2)
    assert mdth.rec_dist(r1, r2).radius < 0.8


def test_node_dist():
    n1 = mbp.from_threshold(lambda x: x[1] >= 0.8, 2).tree
    n2 = mbp.from_threshold(lambda x: x[0] >= 0.6, 2).tree

    d12 = mdth.node_dist(n1, n2)
    assert 0.6 in d12
    assert d12.radius < 0.8


def test_gen_dist_orth():
    p1 = mbp.from_threshold(lambda x: x[1] >= 0.8, 2)
    p2 = mbp.from_threshold(lambda x: x[0] >= 0.6, 2)

    for d12 in fn.take(10, mdth.gen_directed_dists(p1, p2)):
        assert d12.bot - 1e-3 <= 0.6 <= d12.top + 1e-3

    for d21 in fn.take(10, mdth.gen_directed_dists(p2, p1)):
        assert d21.bot - 1e-3 <= 0.8 <= d21.top + 1e-3

    for d in fn.take(10, mdth.gen_dists(p1, p2)):
        assert d.bot - 1e-3 <= 0.8 <= d.top + 1e-3


def test_line_diag_hausdorff(dim=2):
    mbpart = mbp.from_threshold(lambda x: sum(x) >= 0.2, dim)
    d = mbpart.dist(mbpart, tol=1e-1)
    assert d.top <= 1e-1
