from collections import defaultdict
from itertools import product

from multidim_threshold import degenerate

def _compute_responses(rec_set1, rec_set2, metric):
    best_responses = defaultdict(lambda: float('inf'))
    for r1, r2 in product(rec_set1, rec_set2):
        response = metric(r1, r2)
        if response <= best_responses[r1]:
            best_responses[r1] = response
    return best_responses


def directed_hausdorff(rec_set1, rec_set2, *, metric):
    response_map = _compute_responses(rec_set1, rec_set2, metric)
    d = max(response_map.values(), default=0)
    best_moves = {r1 for r1 in rec_set1 if response_map[r1] == d}
    contraints = {r2 for r2 in rec_set2 if any(metric(r1, r2) <= d 
                                               for r1 in best_moves)}
    return d, (best_moves, contraints)


def hausdorff(rec_set1, rec_set2, *, metric):
    d12, req12 = directed_hausdorff(rec_set1, rec_set2, metric=metric) 
    d21, req21 = directed_hausdorff(rec_set2, rec_set1, metric=metric)
    return max(d12, d21), (req12[0] | req21[1], req12[1] | req21[0])


def directed_hausdorff_no_bookkeeping(rec_set1, rec_set2, *, metric):
    return max((min(metric(r1, r2) for r1 in rec_set1)) for r2 in rec_set2)

    
def dist_rec_lowerbound(r1, r2):
    #g1 = lambda x: max(x[2] - x[0] - error, x[3] - x[1] - error, 0)
    g2 = lambda x: max(x[2] - x[1], 0)
    def dist(axis):
        (a,b), (c, d) = axis
        f = sorted([a,b,c,d])
        if set(f[:2]) & set([a, b]) and set(f[:2]) & set([c, d]):
            return 0
        return g2(f)
    return max(map(dist, zip(r1.intervals, r2.intervals)))


def dist_rec_upperbound(r1, r2):
    def dist(axis):
        (a,b), (c, d) = axis
        f = sorted([a,b,c,d])
        return f[-1] - f[0]
    if r1 == r2 and degenerate(r1):
        return 0

    return max(map(dist, zip(r1.intervals, r2.intervals)))

def dist_rec_bounds(r1, r2):
    return dist_rec_lowerbound(r1, r2), dist_rec_upperbound(r1, r2)

def hausdorff_lowerbound(rec_set1, rec_set2):
    return hausdorff(rec_set1, rec_set2, metric=dist_rec_lowerbound)

def hausdorff_upperbound(rec_set1, rec_set2):
    return hausdorff(rec_set1, rec_set2, metric=dist_rec_upperbound)
