import numpy as np
import multidim_threshold as mdt
from collections import namedtuple
import more_itertools

ProjectAlong = namedtuple('ProjectAlong', 'root direc')

new_hi = lambda pi, hi: pi.root + np.multiply(pi.direc, np.true_divide(np.array(hi) - np.array(pi.root), np.array(pi.direc)).min())
midpoint = lambda a, b: np.true_divide(np.array(a) + np.array(b), 2)

def project_along_axes(lo, mid):
    return [project_along_i(lo[:], mid, i) for i in range(len(lo))]

def project_along_i(lo, mid, i):
    lo[i] = mid[i]
    return lo

def proj_key(proj_vec):
    return tuple(map(tuple, proj_vec))

def project_singleLambda(hi, member_oracles, pi):
    # Wasn't sure whether to add diagsearch because it gets decided by the type of oracle right? What is the oracle is
    # boolean but the user wants to do robustness guided search?
    new_high = new_hi(pi, hi)
    rec = mdt.to_rec(pi.root, new_high)
    proj_map = {}
    for oracle in enumerate(member_oracles):
        bool_oracle = isinstance(oracle(rec.bot), bool)
        diagsearch = mdt.binsearch if bool_oracle else mdt.weightedbinsearch
        low, mid, high = diagsearch(rec, oracle)
        proj_map["i"] = mid
    return proj_map

def projections(hi, member_oracles, proj_vectors):
    projections = (project_singleLambda(hi, member_oracles, proj) for proj in proj_vectors)
    return {proj_key(vec): proj for vec, proj in zip(proj_vectors, projections)}


def generate_lambdas(lo, hi, direc, k):
    lambda_0 = ProjectAlong(lo, direc)
    lambdas = [lambda_0]

    new_high = new_hi(lambda_0, hi)
    lambda_mid = [midpoint(lo, new_high)]
    iter = 0

    for iter in range(k):
        projected_points = find_projected_points(lo, lambda_mid)
        # Comment: I used itertools cause the flatten in funcy makes in a single list. basically I need [[[1,2], [2,3]]]
        #  to be [[1,2], [2,3]] and not [1,2,2,3]
        projected_points = list(more_itertools.flatten(projected_points))
        lambda_mid = []
        for i in range(len(projected_points)):
            lambdas.append(ProjectAlong(projected_points[i], direc))
        iter = iter + 1
    return lambdas

def find_projected_points(lo, lambda_mid):
    return [project_along_axes(lo, lambda_mid[i]) for i in range(len(lambda_mid))]












