import numpy as np
import funcy as fn

from monotone_bipartition import rectangles as mdtr  # Interval


def pointwise_hausdorff(xs, ys):
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=float('inf'))

    dXY = pointwise_directed_hausdorff(xs, ys, d=d)
    dYX = pointwise_directed_hausdorff(ys, xs, d=d)
    return max(dXY, dYX)


def pointwise_directed_hausdorff(xs, ys, d):
    return max(min(d(x, y) for y in ys) for x in xs)


def discretized_and_pointwise_hausdorff(recset1, recset2, k=3):
    xs = list(fn.mapcat(lambda r: r.discretize(k), recset1))
    ys = list(fn.mapcat(lambda r: r.discretize(k), recset2))

    error1 = max(r.error + r.shortest_edge for r in recset1)
    error2 = max(r.error + r.shortest_edge for r in recset2)
    error = error1 + error2
    d12 = pointwise_hausdorff(xs, ys)

    return mdtr.Interval(max(d12 - error, 0), d12 + error)
