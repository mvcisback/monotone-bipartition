import numpy as np
import multidim_threshold as mdt
from collections import namedtuple
from lenses import lens
from operator import itemgetter as ig
import funcy as fn

ProjectAlong = namedtuple('ProjectAlong', 'root direc')

new_hi = lambda pi, hi: pi.root + np.multiply(pi.direc, np.true_divide(
    np.array(hi) - np.array(pi.root), np.array(pi.direc)).min())
midpoint = lambda a, b: np.true_divide(np.array(a) + np.array(b), 2)


def project_along_axes(lo, mid):
    return [project_along_i(lo, mid, i) for i in range(len(lo))]


def project_along_i(lo, mid, i):
    return lens(lo)[i].set(mid[i])


def proj_key(proj_vec):
    return tuple(map(tuple, proj_vec))


def learn_diagsearch(member_oracles, bot):
    return [mdt.binsearch if isinstance(oracle(bot), bool) else mdt.weightedbinsearch for oracle in member_oracles]


def project_singleLambda(hi, member_oracles, pi, *, diagsearch=None):
    new_high = new_hi(pi, hi)
    rec = mdt.to_rec(pi.root, new_high)
    if diagsearch is None:
        searches = learn_diagsearch(member_oracles, pi.root)

    searches = learn_diagsearch(member_oracles, pi.root)
    return {i: ig(1)(search(rec, oracle)) for i, (search, oracle) in enumerate(zip(searches, member_oracles))}


def projections(hi, member_oracles, proj_vectors):
    projections = (project_singleLambda(hi, member_oracles, proj)
                   for proj in proj_vectors)
    return {proj_key(vec): proj for vec, proj in zip(proj_vectors, projections)}


def generate_mid_lambdas(lo, hi, direc=None):
    if direc is None:
        direc = hi - lo
    lambda_0 = ProjectAlong(lo, direc)

    new_high = new_hi(lambda_0, hi)
    lambda_mid = [midpoint(lo, new_high)]
    yield lambda_0
    while True:
        projected_points = find_projected_points(lo, lambda_mid)
        projected_points = list(fn.cat(projected_points))
        lambda_mid = [midpoint(point, new_hi(ProjectAlong(
            point, direc), hi)) for point in projected_points]
        yield from (ProjectAlong(p, direc) for p in projected_points)


def find_projected_points(lo, lambda_mid):
    return (project_along_axes(lo, mid) for _, mid in enumerate(lambda_mid))
