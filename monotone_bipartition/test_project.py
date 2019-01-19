import monotone_bipartition as mbp
import monotone_bipartition.search as mbps


def test_proj_line():
    r = mbps.line_intersect(
        func=lambda x: x[0] >= 0.5,
        point=(0.2, 0.2),
        tol=1e-3
    )
    assert (0.5, 0.5) in r
    assert r.shortest_edge < 1e-3

    r = mbps.line_intersect(
        func=lambda x: x[0] >= 0.5,
        point=(1, 1),
        tol=1e-3
    )
    assert (0.5, 0.5) in r
    assert r.shortest_edge < 1e-3

    r = mbps.line_intersect(
        func=lambda x: x[0] + x[1] >= 0.5,
        point=(1, 1),
        tol=1e-3
    )
    assert (0.25, 0.25) in r
    assert r.shortest_edge < 1e-3


def test_proj_lex():
    def oracle(x):
        return x[0] >= 0.5 or x[1] >= 0.5

    r = mbps.lexicographic_opt(
        func=oracle,
        ordering=[(0, True), (1, True)],
        tol=1e-3
    )
    assert abs(r.center[0] - 0.5) < 1e-3
    assert abs(r.center[1] - 0.5) < 1e-3
    assert r.shortest_edge < 1e-3

    r = mbps.lexicographic_opt(
        func=oracle,
        ordering=[(1, True), (0, True)],
        tol=1e-3
    )
    assert abs(r.center[0] - 0.5) < 1e-3
    assert abs(r.center[1] - 0.5) < 1e-3
    assert r.shortest_edge < 1e-3

    r = mbps.lexicographic_opt(
        func=oracle,
        ordering=[(0, False), (1, True)],
        tol=1e-3
    )
    assert abs(r.center[0]) < 1e-3
    assert abs(r.center[1] - 0.5) < 1e-3
    assert r.shortest_edge < 1e-3

    r = mbps.lexicographic_opt(
        func=oracle,
        ordering=[(1, False), (0, True)],
        tol=1e-3
    )
    assert abs(r.center[1]) < 1e-3
    assert abs(r.center[0] - 0.5) < 1e-3
    assert r.shortest_edge < 1e-3

    r = mbps.lexicographic_opt(
        func=lambda x: x[0] > 0.5 and x[1] > 0.5,
        ordering=[(0, True), (1, False)],
        tol=1e-3
    )
    assert abs(r.center[0] - 1) < 1e-3
    assert abs(r.center[1] - 0.5) < 1e-3
    assert r.shortest_edge < 1e-3


def test_bipartition_project_api():
    part = mbp.from_threshold(lambda x: x[0] > 0.5 and x[1] > 0.5, 2)
    r = part.project((1, 1), lexicographic=False)
    assert abs(r.center[0] - 0.5) < 1e-3
    assert abs(r.center[1] - 0.5) < 1e-3

    r = part.project([(0, True), (1, False)], lexicographic=True)
    assert abs(r.center[0] - 1) < 1e-3
    assert abs(r.center[1] - 0.5) < 1e-3
